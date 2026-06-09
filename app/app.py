import streamlit as st
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "models/ticket_classifier.pkl"
)

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Ticket Classifier",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title(
    "🎫 AI Support Ticket Classifier"
)

st.write(
    "Classify support tickets using NLP and Machine Learning."
)

# =========================
# USER INPUT
# =========================

user_input = st.text_area(
    "Enter support ticket text"
)

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict Category"):

    prediction = model.predict(
        [user_input]
    )

    st.success(
        f"Predicted Category: {prediction[0]}"
    )