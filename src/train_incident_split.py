import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

INPUT = Path("data/processed/paper_leak_sentiment_100k_v2.csv")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_PATH = MODEL_DIR / "incident_split_logistic_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "incident_split_tfidf_vectorizer.pkl"
CM_PATH = RESULTS_DIR / "incident_split_confusion_matrix.png"
METRICS_PATH = RESULTS_DIR / "incident_split_metrics.txt"

print("=" * 70)
print("INCIDENT-LEVEL PAPER-LEAK SENTIMENT EXPERIMENT")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT)

df = df.dropna(
    subset=["text", "sentiment", "source_incident_id"]
).copy()

df["text"] = df["text"].astype(str)
df["sentiment"] = df["sentiment"].str.lower().str.strip()
df["source_incident_id"] = df["source_incident_id"].astype(str)

print(f"\nTotal samples: {len(df)}")
print(f"Unique incidents: {df['source_incident_id'].nunique()}")

print("\nClass distribution:")
print(df["sentiment"].value_counts())

# ---------------------------------------------------------
# 2. Incident-level train/test split
# ---------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df["text"],
        df["sentiment"],
        groups=df["source_incident_id"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("\n" + "=" * 70)
print("INCIDENT-LEVEL SPLIT")
print("=" * 70)

print(f"\nTraining samples: {len(train_df)}")
print(f"Testing samples : {len(test_df)}")

print(
    f"Training incidents: "
    f"{train_df['source_incident_id'].nunique()}"
)

print(
    f"Testing incidents : "
    f"{test_df['source_incident_id'].nunique()}"
)

# Verify no incident overlap
overlap = set(
    train_df["source_incident_id"]
).intersection(
    set(test_df["source_incident_id"])
)

print(f"\nIncident overlap: {len(overlap)}")

if len(overlap) != 0:
    raise RuntimeError(
        "ERROR: Incident leakage detected!"
    )

# ---------------------------------------------------------
# 3. Prepare text
# ---------------------------------------------------------

X_train = train_df["text"]
y_train = train_df["sentiment"]

X_test = test_df["text"]
y_test = test_df["sentiment"]

# ---------------------------------------------------------
# 4. TF-IDF
# ---------------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
    max_features=30000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(
    "TF-IDF training shape:",
    X_train_tfidf.shape
)

# ---------------------------------------------------------
# 5. Logistic Regression
# ---------------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

# ---------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------

print("Evaluating...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

report = classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 70)
print("INCIDENT-LEVEL RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(report)

# ---------------------------------------------------------
# 7. Confusion Matrix
# ---------------------------------------------------------

labels = [
    "negative",
    "neutral",
    "positive"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Negative",
        "Neutral",
        "Positive"
    ]
)

disp.plot(ax=ax)

ax.set_title(
    "Incident-Level Logistic Regression"
)

plt.tight_layout()

plt.savefig(
    CM_PATH,
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# 8. Save model
# ---------------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

# ---------------------------------------------------------
# 9. Save metrics
# ---------------------------------------------------------

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "INCIDENT-LEVEL PAPER-LEAK SENTIMENT\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Total samples: {len(df)}\n"
    )

    f.write(
        f"Training samples: {len(train_df)}\n"
    )

    f.write(
        f"Testing samples: {len(test_df)}\n"
    )

    f.write(
        f"Training incidents: "
        f"{train_df['source_incident_id'].nunique()}\n"
    )

    f.write(
        f"Testing incidents: "
        f"{test_df['source_incident_id'].nunique()}\n"
    )

    f.write(
        f"Incident overlap: {len(overlap)}\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1 Score: {f1:.4f}\n\n"
    )

    f.write(report)

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(f"\nModel: {MODEL_PATH}")
print(f"Vectorizer: {VECTORIZER_PATH}")
print(f"Confusion matrix: {CM_PATH}")
print(f"Metrics: {METRICS_PATH}")