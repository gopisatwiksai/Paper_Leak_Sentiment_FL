import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

comparison = pd.DataFrame({
    "Experiment": [
        "Centralized",
        "IID Federated",
        "Non-IID Federated"
    ],
    "Accuracy": [
        0.8182,
        0.8182,
        0.8153
    ],
    "Precision": [
        0.8184,
        0.8183,
        0.8154
    ],
    "Recall": [
        0.8182,
        0.8182,
        0.8153
    ],
    "F1": [
        0.8182,
        0.8182,
        0.8153
    ]
})


# ============================================================
# SAVE COMPARISON TABLE
# ============================================================

comparison_file = (
    "results/final_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

print(
    f"Saved: {comparison_file}"
)


# ============================================================
# GRAPH 1 — ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    comparison["Experiment"],
    comparison["Accuracy"] * 100
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Experiment")
plt.title(
    "Centralized vs Federated Learning Accuracy"
)

plt.ylim(0, 100)

for i, value in enumerate(
    comparison["Accuracy"] * 100
):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

accuracy_file = (
    "results/accuracy_comparison.png"
)

plt.savefig(
    accuracy_file,
    dpi=300
)

plt.close()

print(
    f"Saved: {accuracy_file}"
)


# ============================================================
# GRAPH 2 — ALL METRICS COMPARISON
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

plt.figure(figsize=(10, 6))

x = range(len(comparison))

width = 0.2

for i, metric in enumerate(metrics):

    values = comparison[metric] * 100

    positions = [
        p + (i - 1.5) * width
        for p in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    comparison["Experiment"]
)

plt.ylabel("Score (%)")

plt.xlabel("Experiment")

plt.title(
    "Performance Comparison"
)

plt.ylim(0, 100)

plt.legend()

plt.tight_layout()

metrics_file = (
    "results/metrics_comparison.png"
)

plt.savefig(
    metrics_file,
    dpi=300
)

plt.close()

print(
    f"Saved: {metrics_file}"
)


# ============================================================
# ROUND-BY-ROUND DATA
# ============================================================

round_file = (
    "results/federated_non_iid_rounds_metrics.csv"
)

round_df = pd.read_csv(
    round_file
)


# ============================================================
# GRAPH 3 — FL CONVERGENCE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    round_df["round"],
    round_df["accuracy"] * 100,
    marker="o",
    label="Accuracy"
)

plt.plot(
    round_df["round"],
    round_df["precision"] * 100,
    marker="o",
    label="Precision"
)

plt.plot(
    round_df["round"],
    round_df["recall"] * 100,
    marker="o",
    label="Recall"
)

plt.plot(
    round_df["round"],
    round_df["f1"] * 100,
    marker="o",
    label="F1-score"
)

plt.xlabel("Federated Round")

plt.ylabel("Score (%)")

plt.title(
    "Non-IID Federated Learning Convergence"
)

plt.xticks(
    round_df["round"]
)

plt.ylim(0, 100)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

convergence_file = (
    "results/federated_convergence.png"
)

plt.savefig(
    convergence_file,
    dpi=300
)

plt.close()

print(
    f"Saved: {convergence_file}"
)


# ============================================================
# CLIENT DISTRIBUTION
# ============================================================

client_data = pd.DataFrame({
    "Client": [
        "Client 1",
        "Client 2",
        "Client 3",
        "Client 4",
        "Client 5"
    ],

    "Negative": [
        27.70,
        62.01,
        51.33,
        0.84,
        19.22
    ],

    "Neutral": [
        48.74,
        27.16,
        2.50,
        86.28,
        0.01
    ],

    "Positive": [
        23.57,
        10.83,
        46.16,
        12.88,
        80.77
    ]
})


# ============================================================
# SAVE CLIENT DISTRIBUTION
# ============================================================

client_file = (
    "results/non_iid_client_distribution.csv"
)

client_data.to_csv(
    client_file,
    index=False
)

print(
    f"Saved: {client_file}"
)


# ============================================================
# GRAPH 4 — CLIENT SENTIMENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

x = range(
    len(client_data)
)

width = 0.25

plt.bar(
    [p - width for p in x],
    client_data["Negative"],
    width=width,
    label="Negative"
)

plt.bar(
    x,
    client_data["Neutral"],
    width=width,
    label="Neutral"
)

plt.bar(
    [p + width for p in x],
    client_data["Positive"],
    width=width,
    label="Positive"
)

plt.xticks(
    list(x),
    client_data["Client"]
)

plt.xlabel("Federated Client")

plt.ylabel("Samples (%)")

plt.title(
    "Non-IID Sentiment Distribution Across Clients"
)

plt.legend()

plt.ylim(0, 100)

plt.tight_layout()

client_chart_file = (
    "results/non_iid_client_distribution.png"
)

plt.savefig(
    client_chart_file,
    dpi=300
)

plt.close()

print(
    f"Saved: {client_chart_file}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "RESULT VISUALIZATION COMPLETE"
)

print("=" * 70)

print("\nGenerated files:")

print(
    "1. results/final_model_comparison.csv"
)

print(
    "2. results/accuracy_comparison.png"
)

print(
    "3. results/metrics_comparison.png"
)

print(
    "4. results/federated_convergence.png"
)

print(
    "5. results/non_iid_client_distribution.csv"
)

print(
    "6. results/non_iid_client_distribution.png"
)

print("\nDone.")