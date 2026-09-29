import streamlit as st
import joblib
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Paper Leak Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)

# --------------------------------------------------
# LOAD DEMO MODEL
# --------------------------------------------------

MODEL_PATH = "models/demo_model.pkl"
VECTORIZER_PATH = "models/demo_tfidf.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 Paper Leak Sentiment Analysis")

st.subheader("Using Federated Learning")

st.write(
    "Enter a paper-leak-related comment below to classify "
    "its sentiment using the trained sentiment analysis model."
)

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("FL Clients", "5")

with col2:
    st.metric("Training Method", "FedAvg")

with col3:
    st.metric("Research Model", "TF-IDF + Linear")

st.divider()

# --------------------------------------------------
# INPUT
# --------------------------------------------------

text = st.text_area(
    "Enter a comment:",
    placeholder="Example: The paper leak has caused serious problems for students.",
    height=120
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not text.strip():
        st.warning("Please enter a comment.")
    else:

        # Transform text
        X = vectorizer.transform([text])

        # Prediction
        prediction = model.predict(X)[0]

        # Probabilities
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_

        probability_dict = dict(zip(classes, probabilities))

        confidence = float(np.max(probabilities)) * 100

        # --------------------------------------------------
        # DISPLAY PREDICTION
        # --------------------------------------------------

        st.markdown("### Prediction")

        if prediction == "negative":
            st.error("🔴 Negative")
        elif prediction == "positive":
            st.success("🟢 Positive")
        else:
            st.info("⚪ Neutral")

        st.metric(
            "Model Probability",
            f"{confidence:.2f}%"
        )

        # --------------------------------------------------
        # PROBABILITIES
        # --------------------------------------------------

        st.markdown("### Class Probabilities")

        negative = probability_dict.get("negative", 0) * 100
        neutral = probability_dict.get("neutral", 0) * 100
        positive = probability_dict.get("positive", 0) * 100

        st.write(f"**Negative:** {negative:.2f}%")
        st.progress(float(negative / 100))

        st.write(f"**Neutral:** {neutral:.2f}%")
        st.progress(float(neutral / 100))

        st.write(f"**Positive:** {positive:.2f}%")
        st.progress(float(positive / 100))

        # --------------------------------------------------
        # CONFIDENCE INTERPRETATION
        # --------------------------------------------------

        if confidence >= 70:
            st.success("High model probability")
        elif confidence >= 50:
            st.warning("Moderate model probability")
        else:
            st.warning("Low model probability — the model is uncertain.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Paper Leak Sentiment Analysis Using Federated Learning"
)

st.caption(
    "5 simulated clients • FedAvg • PyTorch • TF-IDF"
)