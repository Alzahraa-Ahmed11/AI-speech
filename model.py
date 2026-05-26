"""
model.py — Pronunciation Scoring Model (Anti-Collapse Edition)

ROOT CAUSE ANALYSIS — WHY THE ORIGINAL COLLAPSED
═════════════════════════════════════════════════

The collapse (pred_std ≈ 0, AUC ≈ 0.5) has FOUR compounding root causes:

1. SCORING HEAD: DEAD LAYERNORM + DEAD ACTIVATIONS AFTER L2-NORMALISED INPUT
   The original head receives 4×embed_dim input built from L2-normalised
   embeddings. At init, all weights are small random → all hidden neurons
   near zero → LayerNorm maps them to N(0,1) → GELU kills about 50%.
   The second LayerNorm+GELU kills another 50%, leaving the final Linear
   almost no gradient to propagate back. The head collapses to near-constant
   output within 1-2 batches.

   FIX: Replace LayerNorm+GELU stack with ELU (no dead-neuron problem),
   add a residual bypass so gradient has a direct path, use Kaiming He init.

2. COS_WEIGHT=0.2 + cos_margin=0.3 → ENCODER GETS NEAR-ZERO GRADIENT
   CosineEmbeddingLoss for NEGATIVE pairs = max(0, cos - margin).
   Random encoder → cos≈0 → loss = max(0, 0 - 0.3) = 0 for negatives.
   Negatives provide ZERO gradient to the encoder from step 1.
   Positives get some gradient, but with cos_weight=0.2 the signal is tiny.

   FIX: cos_weight=1.5, cos_margin=0.0 (loss is non-zero from the very first step).

3. ENTIRE WAV2VEC2 TRANSFORMER TRAINABLE FROM EPOCH 1 AT LR=1e-4
   wav2vec2-base has 94M parameters. Training it simultaneously with the
   head (only ~460K params) at the same LR destroys the pretrained features
   before the head establishes any useful signal.

   FIX: Freeze the transformer for warmup_epochs (phase 1). After warmup,
   unfreeze with 20× smaller LR via differential optimizer param groups.

4. COSINE ANNEALING STARTS DECAYING IMMEDIATELY
   LR drops from epoch 1, reducing signal during the critical warmup phase.
   FIX: Flat LR for warmup_epochs, then cosine decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Wav2Vec2PronunciationEncoder


class AttentivePooling(nn.Module):
    """
    Collapse a sequence of frame-level hidden states into one fixed-size vector.
    Learns a scalar importance score per frame, then returns the weighted sum.
    Padding frames are masked to -inf before softmax so they contribute zero weight.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x    : [batch, time, hidden_dim]
        mask : [batch, time] bool  — True = real frame, False = padding
        Returns [batch, hidden_dim]
        """
        scores = self.attn(x).squeeze(-1)  # [batch, time]
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=1.0 / scores.shape[-1])
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # [batch, hidden_dim]


class _LegacyCnnPronunciationEncoder(nn.Module):
    """
    Encode a raw waveform into an L2-normalised pronunciation embedding.

    Changes vs original:
    - CNN feature extractor is always frozen (unchanged).
    - Transformer is frozen initially; call unfreeze_transformer() after warmup.
    - Projection MLP is unchanged (not a collapse source).
    """

    def __init__(self, embed_dim: int = 256, freeze_feature_extractor: bool = True):
        super().__init__()

        self.n_fft = 400
        self.hop_length = 160
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ELU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Linear(256, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _freeze_transformer(self):
        return None

    def unfreeze_transformer(self):
        """Unfreeze the transformer for phase-2 fine-tuning."""
        return None

    def transformer_parameters(self):
        """Return transformer params for differential LR scheduling."""
        return []

    def head_parameters(self):
        """Return projection head + pooling params (trained at full LR)."""
        return list(self.features.parameters()) + list(self.proj.parameters())

    def forward(
        self, waveform: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        waveform       : [batch, samples]  raw PCM at 16 kHz
        attention_mask : [batch, samples] bool
        Returns [batch, embed_dim]  L2-normalised.
        """
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(waveform.device),
            return_complex=True,
            center=True,
        )
        spec = torch.log1p(spec.abs()).unsqueeze(1)  # [B, 1, freq, frames]
        feat = self.features(spec)
        avg_pool = feat.mean(dim=(-2, -1))
        max_pool = feat.amax(dim=(-2, -1))
        pooled = torch.cat([avg_pool, max_pool], dim=-1)
        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=-1)


PronunciationEncoder = Wav2Vec2PronunciationEncoder


class ScoringHead(nn.Module):
    """
    Compare two L2-normalised embeddings and output a raw pronunciation logit.

    KEY FIXES vs original:
    - LayerNorm+GELU replaced with ReLU; Kaiming He init on all linear weights.

    Input: [ref_emb, child_emb, |ref-child|, ref*child]  → 4×embed_dim features
    Output: unbounded scalar logit per pair.
      Training  → BCEWithLogitsLoss(logit, target)
      Inference → score_from_embeddings() applies sigmoid → [0, 1] probability
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        in_dim = embed_dim * 4
        mid    = embed_dim * 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mid, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, ref_emb: torch.Tensor, child_emb: torch.Tensor) -> torch.Tensor:
        """
        ref_emb, child_emb : [batch, embed_dim]  L2-normalised
        Returns [batch]  raw logit.
        """
        ref_emb   = F.normalize(ref_emb,   p=2, dim=-1, eps=1e-8)
        child_emb = F.normalize(child_emb, p=2, dim=-1, eps=1e-8)
        diff = torch.abs(ref_emb - child_emb)
        prod = ref_emb * child_emb
        combined = torch.cat([ref_emb, child_emb, diff, prod], dim=-1)  # [B, 4D]
        return self.net(combined).squeeze(-1)  # [batch]


class PronunciationScoringModel(nn.Module):
    """
    End-to-end Siamese pronunciation scoring model.

    Training protocol:
      Phase 1 — warmup: transformer frozen, head + projection trained at full LR.
      Phase 2 — fine-tune: call model.unfreeze_encoder(), use differential LR
                (transformer LR = head LR × 0.05).

    Training path: raw_logit_from_embeddings / logit_from_embeddings → raw logit
      (used with BCEWithLogitsLoss in train.py).
    Inference path: score_from_embeddings → probability in [0, 1]
      (temperature + margin applied pre-sigmoid; multiply by 100 for a 0-100 score).
    """

    def __init__(self, embed_dim: int = 256, freeze_feature_extractor: bool = True):
        super().__init__()
        self.encoder   = Wav2Vec2PronunciationEncoder(embed_dim, freeze_feature_extractor=freeze_feature_extractor)
        self.scorer    = ScoringHead(embed_dim)
        self.embed_dim = embed_dim
        self.use_scoring_head = True
        self.score_temperature = 1.35
        self.cosine_margin = 0.72
        self.cosine_penalty_weight = 3.0
        self.fallback_margin = 0.94
        self.fallback_temperature = 0.04

    def unfreeze_encoder(self, trainable_layers: int = None):
        """Unfreeze the wav2vec2 transformer for phase-2 fine-tuning."""
        self.encoder.unfreeze_transformer(trainable_layers=trainable_layers)

    def get_param_groups(self, head_lr: float, encoder_lr_scale: float = 0.05):
        """
        Return optimizer param groups with differential learning rates.

        head_lr          : full LR for scoring head + projection head + pooling
        encoder_lr_scale : fraction of head_lr for the transformer (default 5%)

        Use this after unfreeze_encoder() to avoid destroying pretrained features.
        """
        head_params = (
            list(self.encoder.head_parameters())
            + list(self.scorer.parameters())
        )
        transformer_params = self.encoder.transformer_parameters()
        return [
            {"params": head_params,        "lr": head_lr},
            {"params": transformer_params, "lr": head_lr * encoder_lr_scale},
        ]

    def forward(
        self,
        ref_waveform:   torch.Tensor,
        child_waveform: torch.Tensor,
        ref_mask:       torch.Tensor = None,
        child_mask:     torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns [batch] raw logit."""
        ref_emb   = self.encoder(ref_waveform,   ref_mask)
        child_emb = self.encoder(child_waveform, child_mask)
        return self.raw_logit_from_embeddings(ref_emb, child_emb)

    def encode(self, waveform: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Encode one waveform to an L2-normalised embedding."""
        return self.encoder(waveform, mask)

    def raw_logit_from_embeddings(
        self, ref_emb: torch.Tensor, child_emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the trainable MLP logit. Used by the training objective."""
        ref_emb   = F.normalize(ref_emb,   p=2, dim=-1, eps=1e-8)
        child_emb = F.normalize(child_emb, p=2, dim=-1, eps=1e-8)
        return self.scorer(ref_emb, child_emb)

    def score_from_embeddings(
        self, ref_emb: torch.Tensor, child_emb: torch.Tensor
    ) -> torch.Tensor:
        """Return calibrated pronunciation probability in [0, 1]."""
        ref_emb   = F.normalize(ref_emb,   p=2, dim=-1, eps=1e-8)
        child_emb = F.normalize(child_emb, p=2, dim=-1, eps=1e-8)
        cos = F.cosine_similarity(ref_emb, child_emb, dim=-1, eps=1e-8)
        if not self.use_scoring_head:
            return torch.sigmoid((cos - self.fallback_margin) / self.fallback_temperature)

        raw = self.raw_logit_from_embeddings(ref_emb, child_emb)
        margin_penalty = F.relu(self.cosine_margin - cos) * self.cosine_penalty_weight
        return torch.sigmoid((raw / self.score_temperature) - margin_penalty)

    def logit_from_embeddings(
        self, ref_emb: torch.Tensor, child_emb: torch.Tensor
    ) -> torch.Tensor:
        """Alias for score_from_embeddings() — train.py compatibility."""
        return self.raw_logit_from_embeddings(ref_emb, child_emb)

    @torch.no_grad()
    def predict_score(
        self,
        ref_waveform:   torch.Tensor,
        child_waveform: torch.Tensor,
    ) -> float:
        """Inference helper: 0-100 pronunciation score. Applies sigmoid internally."""
        self.eval()
        if ref_waveform.dim() == 1:
            ref_waveform   = ref_waveform.unsqueeze(0)
        if child_waveform.dim() == 1:
            child_waveform = child_waveform.unsqueeze(0)
        ref_emb = self.encode(ref_waveform)
        child_emb = self.encode(child_waveform)
        prob = self.score_from_embeddings(ref_emb, child_emb)
        return float(prob.mul(100.0).clamp(0.0, 100.0).item())
