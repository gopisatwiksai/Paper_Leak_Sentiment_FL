import os
import random
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed/paper_leak_sentiment_100k.csv"

RESULT_FILE = "results/federated_non_iid_v1_results.txt"
PREDICTION_FILE = "results/federated_non_iid_v1_predictions.csv"
MODEL_FILE = "models/federated_non_iid_v1_global_model.pt"
VECTORIZER_FILE = "models/federated_non_iid_v1_tfidf.pkl"

RANDOM_STATE = 42

NUM_CLIENTS = 5
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2

LEARNING_RATE = 0.05
BATCH_SIZE = 256

# Smaller alpha = more heterogeneous clients.
# 0.5 gives noticeable but still usable non-IID distributions.
DIRICHLET_ALPHA = 0.5

DEVICE = torch.device("cpu")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# MODEL
# ============================================================

class SentimentModel(nn.Module):

    def __init__(self, input_size, num_classes=3):

        super().__init__()

        self.linear = nn.Linear(
            input_size,
            num_classes
        )

    def forward(self, x):

        return self.linear(x)


# ============================================================
# FEDAVG
# ============================================================

def fedavg(client_states, client_sizes):

    total_samples = sum(client_sizes)

    global_state = {}

    for key in client_states[0].keys():

        weighted_sum = torch.zeros_like(
            client_states[0][key]
        )

        for state, size in zip(
            client_states,
            client_sizes
        ):

            weight = size / total_samples

            weighted_sum += (
                state[key] * weight
            )

        global_state[key] = weighted_sum

    return global_state


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_client(
    global_state,
    X_client,
    y_client,
    input_size
):

    model = SentimentModel(
        input_size=input_size,
        num_classes=3
    )

    model.load_state_dict(
        global_state
    )

    model.to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(
        X_client.toarray(),
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_client,
        dtype=torch.long
    )

    dataset_size = len(y_tensor)

    model.train()

    for epoch in range(LOCAL_EPOCHS):

        indices = torch.randperm(
            dataset_size
        )

        for start in range(
            0,
            dataset_size,
            BATCH_SIZE
        ):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            X_batch = X_tensor[
                batch_indices
            ]

            y_batch = y_tensor[
                batch_indices
            ]

            optimizer.zero_grad()

            output = model(
                X_batch
            )

            loss = criterion(
                output,
                y_batch
            )

            loss.backward()

            optimizer.step()

    return (
        model.state_dict(),
        dataset_size
    )


# ============================================================
# NON-IID CLIENT DISTRIBUTION
# ============================================================

def create_non_iid_clients(
    y,
    num_clients,
    alpha,
    random_state
):

    rng = np.random.default_rng(
        random_state
    )

    num_classes = 3

    client_indices = [
        []
        for _ in range(num_clients)
    ]

    print("\nCreating Non-IID distribution...")

    for class_id in range(num_classes):

        class_indices = np.where(
            y == class_id
        )[0]

        rng.shuffle(
            class_indices
        )

        # Generate client proportions for this class.
        proportions = rng.dirichlet(
            np.repeat(
                alpha,
                num_clients
            )
        )

        # Convert proportions to sample counts.
        counts = (
            proportions *
            len(class_indices)
        ).astype(int)

        # Correct rounding difference.
        difference = (
            len(class_indices)
            - counts.sum()
        )

        if difference > 0:

            largest_clients = np.argsort(
                proportions
            )[::-1]

            for i in range(difference):

                counts[
                    largest_clients[i % num_clients]
                ] += 1

        start = 0

        for client_id, count in enumerate(
            counts
        ):

            end = start + count

            client_indices[
                client_id
            ].extend(
                class_indices[
                    start:end
                ].tolist()
            )

            start = end

    # Shuffle each client's complete dataset.
    for client_id in range(num_clients):

        rng.shuffle(
            client_indices[client_id]
        )

    return client_indices


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("NON-IID FEDERATED LEARNING — PAPER LEAK SENTIMENT")
print("=" * 70)

print(
    f"\nDevice: {DEVICE}"
)

print(
    f"Clients: {NUM_CLIENTS}"
)

print(
    f"Rounds: {NUM_ROUNDS}"
)

print(
    f"Local epochs: {LOCAL_EPOCHS}"
)

print(
    f"Dirichlet alpha: {DIRICHLET_ALPHA}"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE
)

required_columns = [
    "text",
    "sentiment",
    "source_incident_id"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )

df = df.dropna(
    subset=required_columns
).copy()

df["text"] = df[
    "text"
].astype(str)

df["sentiment"] = (
    df["sentiment"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df["source_incident_id"] = (
    df["source_incident_id"]
    .astype(str)
    .str.strip()
)

print(
    f"Total samples: {len(df):,}"
)

print(
    f"Unique incidents: "
    f"{df['source_incident_id'].nunique()}"
)


# ============================================================
# INCIDENT-LEVEL TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("INCIDENT-LEVEL TRAIN / TEST SPLIT")
print("=" * 70)

incident_ids = sorted(
    df["source_incident_id"].unique()
)

random.seed(
    RANDOM_STATE
)

random.shuffle(
    incident_ids
)

train_incident_count = round(
    len(incident_ids) * 0.80
)

train_incidents = set(
    incident_ids[:train_incident_count]
)

test_incidents = set(
    incident_ids[train_incident_count:]
)

train_df = df[
    df["source_incident_id"].isin(
        train_incidents
    )
].copy()

test_df = df[
    df["source_incident_id"].isin(
        test_incidents
    )
].copy()

overlap = (
    train_incidents &
    test_incidents
)

print(
    f"\nTraining incidents: "
    f"{len(train_incidents)}"
)

print(
    f"Test incidents: "
    f"{len(test_incidents)}"
)

print(
    f"Training samples: "
    f"{len(train_df):,}"
)

print(
    f"Test samples: "
    f"{len(test_df):,}"
)

print(
    f"Incident overlap: "
    f"{len(overlap)}"
)

if overlap:

    raise ValueError(
        "Incident leakage detected!"
    )


# ============================================================
# LABEL ENCODING
# ============================================================

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

reverse_label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

train_df["label"] = train_df[
    "sentiment"
].map(label_map)

test_df["label"] = test_df[
    "sentiment"
].map(label_map)

if train_df["label"].isna().any():

    raise ValueError(
        "Unknown training sentiment found."
    )

if test_df["label"].isna().any():

    raise ValueError(
        "Unknown test sentiment found."
    )


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    max_features=50_000
)

X_train = vectorizer.fit_transform(
    train_df["text"]
)

X_test = vectorizer.transform(
    test_df["text"]
)

y_train = train_df[
    "label"
].to_numpy()

y_test = test_df[
    "label"
].to_numpy()

print(
    f"\nTraining matrix: "
    f"{X_train.shape}"
)

print(
    f"Test matrix: "
    f"{X_test.shape}"
)


# ============================================================
# CREATE NON-IID CLIENTS
# ============================================================

print("\n" + "=" * 70)
print("NON-IID CLIENT DISTRIBUTION")
print("=" * 70)

client_indices = create_non_iid_clients(
    y_train,
    NUM_CLIENTS,
    DIRICHLET_ALPHA,
    RANDOM_STATE
)

clients = []

for client_id, indices in enumerate(
    client_indices,
    start=1
):

    indices = np.array(
        indices,
        dtype=int
    )

    X_client = X_train[
        indices
    ]

    y_client = y_train[
        indices
    ]

    clients.append(
        (
            X_client,
            y_client
        )
    )

    counts = np.bincount(
        y_client,
        minlength=3
    )

    total = len(y_client)

    print(
        f"\nClient {client_id}"
    )

    print(
        f"  Total: {total:,}"
    )

    print(
        f"  Negative: {counts[0]:,} "
        f"({counts[0] / total * 100:.2f}%)"
    )

    print(
        f"  Neutral : {counts[1]:,} "
        f"({counts[1] / total * 100:.2f}%)"
    )

    print(
        f"  Positive: {counts[2]:,} "
        f"({counts[2] / total * 100:.2f}%)"
    )


# ============================================================
# INITIAL GLOBAL MODEL
# ============================================================

input_size = X_train.shape[1]

global_model = SentimentModel(
    input_size=input_size,
    num_classes=3
)

global_state = (
    global_model.state_dict()
)


# ============================================================
# FEDERATED TRAINING
# ============================================================

print("\n" + "=" * 70)
print("NON-IID FEDERATED TRAINING")
print("=" * 70)

for round_number in range(
    1,
    NUM_ROUNDS + 1
):

    print(
        f"\n--- Federated Round "
        f"{round_number}/{NUM_ROUNDS} ---"
    )

    client_states = []
    client_sizes = []

    for client_id, (
        X_client,
        y_client
    ) in enumerate(
        clients,
        start=1
    ):

        local_state, local_size = (
            train_client(
                global_state,
                X_client,
                y_client,
                input_size
            )
        )

        client_states.append(
            local_state
        )

        client_sizes.append(
            local_size
        )

        print(
            f"Client {client_id} "
            f"local training complete "
            f"({local_size:,} samples)"
        )

    global_state = fedavg(
        client_states,
        client_sizes
    )

    print(
        "FedAvg aggregation complete."
    )


# ============================================================
# GLOBAL MODEL
# ============================================================

global_model.load_state_dict(
    global_state
)

global_model.to(DEVICE)

global_model.eval()


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("NON-IID GLOBAL MODEL EVALUATION")
print("=" * 70)

X_test_tensor = torch.tensor(
    X_test.toarray(),
    dtype=torch.float32
)

with torch.no_grad():

    logits = global_model(
        X_test_tensor
    )

    predictions = torch.argmax(
        logits,
        dim=1
    ).cpu().numpy()

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

target_names = [
    "negative",
    "neutral",
    "positive"
]

report = classification_report(
    y_test,
    predictions,
    target_names=target_names,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions
)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(matrix)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)

prediction_df = test_df[
    [
        "id",
        "text",
        "sentiment",
        "source_incident_id"
    ]
].copy()

prediction_df[
    "predicted_sentiment"
] = [
    reverse_label_map[i]
    for i in predictions
]

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# ============================================================
# SAVE VECTORIZER
# ============================================================

with open(
    VECTORIZER_FILE,
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f
    )


# ============================================================
# SAVE GLOBAL MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": global_state,
        "input_size": input_size,
        "num_classes": 3,
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dirichlet_alpha": DIRICHLET_ALPHA
    },
    MODEL_FILE
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_text = f"""
PAPER LEAK SENTIMENT ANALYSIS
NON-IID FEDERATED LEARNING V1
=============================

Dataset:
{DATA_FILE}

Total samples:
{len(df):,}

Unique incidents:
{df['source_incident_id'].nunique()}

Training incidents:
{len(train_incidents)}

Test incidents:
{len(test_incidents)}

Incident overlap:
{len(overlap)}

Training samples:
{len(train_df):,}

Test samples:
{len(test_df):,}

Federated clients:
{NUM_CLIENTS}

Federated rounds:
{NUM_ROUNDS}

Local epochs:
{LOCAL_EPOCHS}

Learning rate:
{LEARNING_RATE}

Dirichlet alpha:
{DIRICHLET_ALPHA}

TF-IDF:
ngram_range = (1, 2)
min_df = 3
max_df = 0.95
sublinear_tf = True
max_features = 50,000

Model:
PyTorch Linear Softmax Classifier

Aggregation:
Federated Averaging (FedAvg)

Data distribution:
Non-IID using Dirichlet allocation

Device:
{DEVICE}

RESULTS

Accuracy:
{accuracy:.4f}

Precision:
{precision:.4f}

Recall:
{recall:.4f}

F1-score:
{f1:.4f}

Classification Report:
{report}

Confusion Matrix:
{matrix}
"""

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        result_text
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("NON-IID FEDERATED LEARNING COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to:"
    f"\n{RESULT_FILE}"
)

print(
    f"\nPredictions saved to:"
    f"\n{PREDICTION_FILE}"
)

print(
    f"\nGlobal model saved to:"
    f"\n{MODEL_FILE}"
)

print(
    f"\nTF-IDF vectorizer saved to:"
    f"\n{VECTORIZER_FILE}"
)

print("\nDone.")