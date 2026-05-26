import json
import matplotlib.pyplot as plt
import numpy as np

with open("logs/history.json", "r") as f:
    data = json.load(f)

epochs = [d["epoch"] for d in data]

# =========================
# 1. Loss Curve
# =========================
plt.figure()
plt.plot(epochs, [d["train_loss"] for d in data], label="Train Loss")
plt.plot(epochs, [d["val_loss"] for d in data], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png", dpi=300)
plt.close()

# =========================
# 2. AUC Curve
# =========================
plt.figure()
plt.plot(epochs, [d["train_auc"] for d in data], label="Train AUC")
plt.plot(epochs, [d["val_auc"] for d in data], label="Val AUC")
plt.title("AUC Curve")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid()
plt.savefig("auc_curve.png", dpi=300)
plt.close()

# =========================
# 3. Accuracy Curve
# =========================
plt.figure()
plt.plot(epochs, [d["val_acc"] for d in data], label="Val Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

print("Done: images saved (loss_curve.png, auc_curve.png, accuracy_curve.png)")