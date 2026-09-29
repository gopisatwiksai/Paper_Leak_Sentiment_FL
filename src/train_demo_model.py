import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA = "data/processed/paper_leak_sentiment_100k.csv"

df = pd.read_csv(DATA)
df["text"] = df["text"].fillna("").astype(str)
df["sentiment"] = df["sentiment"].astype(str).str.lower()

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["sentiment"],
    random_state=42
)

features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            max_features=60000
        )
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=60000
        )
    )
])

X_train = features.fit_transform(train_df["text"])
X_test = features.transform(test_df["text"])

model = LogisticRegression(
    max_iter=1500,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, train_df["sentiment"])

pred = model.predict(X_test)

print("=" * 60)
print("DEMO MODEL BENCHMARK")
print("=" * 60)
print(f"Training samples: {len(train_df)}")
print(f"Test samples:     {len(test_df)}")
print(f"Features:         {X_train.shape[1]}")
print(f"Accuracy:         {accuracy_score(test_df['sentiment'], pred):.4f}")
print()
print(classification_report(test_df["sentiment"], pred))

demo_texts = [
    "Students are angry.",
    "Students are happy.",
    "An investigation was announced.",
    "The paper leak has caused serious problems for students.",
    "Students appreciated the authorities for taking quick action."
]

X_demo = features.transform(demo_texts)
probs = model.predict_proba(X_demo)
classes = model.classes_

print("=" * 60)
print("DEMO SENTENCES")
print("=" * 60)

for text, probability in zip(demo_texts, probs):
    order = probability.argsort()[::-1]

    print()
    print(text)

    for i in order:
        print(f"  {classes[i]:8s}: {probability[i] * 100:.2f}%")

joblib.dump(model, "models/demo_model.pkl")
joblib.dump(features, "models/demo_tfidf.pkl")

print()
print("Saved:")
print("models/demo_model.pkl")
print("models/demo_tfidf.pkl")
