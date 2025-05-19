import pandas as pd
import matplotlib.pyplot as plt

# Data from the user's SVM grid search
data = {
    "Kernel": ["linear"] * 5 + ["rbf"] * 5 + ["poly"] * 5,
    "C": [0.01, 0.1, 1.0, 10.0, 100.0] * 3,
    "Accuracy": [
        0.8537, 0.9055, 0.9151, 0.9156, 0.9213,
        0.8403, 0.9084, 0.9166, 0.9176, 0.9003,
        0.8475, 0.9107, 0.9114, 0.9055, 0.9025
    ],
    "F1-score": [
        0.8466, 0.9043, 0.9130, 0.9110, 0.9166,
        0.8251, 0.9058, 0.9113, 0.9148, 0.8940,
        0.8388, 0.9071, 0.9080, 0.9035, 0.8992
    ]
}


df = pd.DataFrame(data)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

# Plot Accuracy
for kernel in df["Kernel"].unique():
    subset = df[df["Kernel"] == kernel].sort_values(by="C")
    axs[0].plot(subset["C"], subset["Accuracy"], marker='o', label=kernel)
axs[0].set_xscale("log")
axs[0].set_title("Accuracy vs C")
axs[0].set_xlabel("C (log scale)")
axs[0].set_ylabel("Score")
axs[0].grid(True)
axs[0].legend()

# Plot F1-score
for kernel in df["Kernel"].unique():
    subset = df[df["Kernel"] == kernel].sort_values(by="C")
    axs[1].plot(subset["C"], subset["F1-score"], marker='o', label=kernel)

axs[1].set_xscale("log")
axs[1].set_title("F1-score vs C")
axs[1].set_xlabel("C (log scale)")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

