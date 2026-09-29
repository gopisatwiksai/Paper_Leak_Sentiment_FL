import os
import pickle
import torch
import torch.nn as nn
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "federated_learning_v1_global_model.pt"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "federated_learning_v1_tfidf.pkl"
)


# ============================================================
# MODEL
# ============================================================

class SentimentModel(nn.Module):

    def __init__(self, input_size, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Load TF-IDF
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    input_size = len(vectorizer.get_feature_names_out())

    # Create model
    model = SentimentModel(input_size)

    # Load FL checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, vectorizer


# ============================================================
# PREDICTION
# ============================================================

def predict_sentiment(text, model, vectorizer):

    vector = vectorizer.transform([text])

    features = torch.tensor(
        vector.toarray(),
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = model(features)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    labels = [
        "Negative",
        "Neutral",
        "Positive"
    ]

    sentiment = labels[predicted_class]

    confidence = (
        probabilities[0][predicted_class].item()
        * 100
    )

    return sentiment, confidence, probabilities[0]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Paper Leak Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Paper Leak Sentiment Analysis")

st.subheader("Using Federated Learning")

st.write(
    "Enter a paper-leak-related comment below "
    "to classify its sentiment using the trained "
    "Federated Learning global model."
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "FL Clients",
        "5"
    )

with col2:
    st.metric(
        "Training Method",
        "FedAvg"
    )

with col3:
    st.metric(
        "Model",
        "TF-IDF + Linear"
    )


st.divider()


# ============================================================
# TEXT INPUT
# ============================================================

text = st.text_area(
    "Enter your comment:",
    height=150,
    placeholder=(
        "Example: The paper leak has seriously "
        "affected thousands of students..."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "Please enter a comment first."
        )

    else:

        try:

            model, vectorizer = load_model()

            sentiment, confidence, probabilities = (
                predict_sentiment(
                    text,
                    model,
                    vectorizer
                )
            )

            st.divider()

            st.subheader("Prediction")

            # Sentiment display
            if sentiment == "Negative":

                st.error(
                    f"🔴 {sentiment}"
                )

            elif sentiment == "Positive":

                st.success(
                    f"🟢 {sentiment}"
                )

            else:

                st.info(
                    f"🔵 {sentiment}"
                )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.divider()

            # Probability breakdown
            st.subheader(
                "Class Probabilities"
            )

            labels = [
                "Negative",
                "Neutral",
                "Positive"
            ]

            for label, probability in zip(
                labels,
                probabilities
            ):

                value = probability.item()

                st.write(
                    f"**{label}:** "
                    f"{value * 100:.2f}%"
                )

                st.progress(
                    min(value, 1.0)
                )

        except Exception as e:

            st.error(
                "An error occurred while "
                "loading the model."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Paper Leak Sentiment Analysis Using Federated Learning"
)

st.caption(
    "5 simulated clients • FedAvg • PyTorch • TF-IDF"
)