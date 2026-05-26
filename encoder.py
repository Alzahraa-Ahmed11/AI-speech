import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


DEFAULT_SPEECH_MODEL = os.getenv("SPEECH_ENCODER_MODEL", "facebook/wav2vec2-base")


class Wav2Vec2PronunciationEncoder(nn.Module):
    """
    Pretrained self-supervised speech encoder for pronunciation embeddings.

    The HuggingFace model returns hidden states over time. We mean-pool those
    states, project to the repo's embedding size, and L2-normalise the result
    so cosine similarity remains stable for API diagnostics and training.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        model_name: str = DEFAULT_SPEECH_MODEL,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.speech_model = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.speech_model.config.hidden_size)

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, embed_dim),
        )

        if freeze_feature_extractor:
            self._freeze_conv_feature_extractor()

        self._freeze_transformer()

    def _freeze_conv_feature_extractor(self) -> None:
        if hasattr(self.speech_model, "freeze_feature_encoder"):
            self.speech_model.freeze_feature_encoder()
        elif hasattr(self.speech_model, "freeze_feature_extractor"):
            self.speech_model.freeze_feature_extractor()

    def _freeze_transformer(self) -> None:
        for param in self.speech_model.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self, trainable_layers: Optional[int] = None) -> None:
        for param in self.speech_model.parameters():
            param.requires_grad = trainable_layers is None

        layers = getattr(getattr(self.speech_model, "encoder", None), "layers", None)
        if trainable_layers is not None and layers is None:
            for param in self.speech_model.parameters():
                param.requires_grad = True

        if trainable_layers is not None and layers is not None:
            trainable_layers = max(0, min(int(trainable_layers), len(layers)))
            if trainable_layers > 0:
                for layer in layers[-trainable_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            for module_name in ("feature_projection", "encoder.layer_norm"):
                module = self.speech_model
                for part in module_name.split("."):
                    module = getattr(module, part, None)
                    if module is None:
                        break
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True

        self._freeze_conv_feature_extractor()

    def transformer_parameters(self):
        return [p for p in self.speech_model.parameters() if p.requires_grad]

    def head_parameters(self):
        return list(self.proj.parameters())

    def _feature_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.long)
        try:
            mask = self.speech_model._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
        except Exception:
            mask = F.interpolate(
                attention_mask[:, None].float(),
                size=hidden_states.shape[1],
                mode="nearest",
            ).squeeze(1).bool()
        return mask.to(device=hidden_states.device, dtype=torch.bool)

    @staticmethod
    def _mean_pool(
        hidden_states: torch.Tensor,
        feature_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if feature_mask is None:
            return hidden_states.mean(dim=1)

        weights = feature_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        summed = (hidden_states * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        waveform: [batch, samples] raw PCM at 16 kHz
        attention_mask: [batch, samples] where 1/True marks real audio
        returns: [batch, embed_dim] L2-normalised embeddings
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        device = next(self.parameters()).device
        waveform = waveform.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)

        # When the speech model is fully frozen, skip gradient bookkeeping for efficiency.
        # During phase-2 fine-tuning (unfreeze_transformer called), gradients flow normally.
        speech_needs_grad = any(p.requires_grad for p in self.speech_model.parameters())
        if speech_needs_grad:
            outputs = self.speech_model(
                input_values=waveform,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                outputs = self.speech_model(
                    input_values=waveform,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )

        hidden_states = outputs.last_hidden_state
        feature_mask = self._feature_attention_mask(hidden_states, attention_mask)
        pooled = self._mean_pool(hidden_states, feature_mask)
        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=-1, eps=1e-8)
