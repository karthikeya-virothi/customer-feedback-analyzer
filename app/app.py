import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="💬",
    layout="centered"
)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():

    model_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    return model, tokenizer, device


model, tokenizer, device = load_model()

# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------
def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    if prediction == 1:
        return "Positive 😊"
    else:
        return "Negative 😡"


# ---------------------------------
# UI
# ---------------------------------
st.title("💬 Customer Feedback Analyzer")

st.markdown("### NLP + Transformers Based Sentiment Analysis")

st.write(
    "Analyze customer reviews using Machine Learning, NLP, and Hugging Face Transformers."
)

user_input = st.text_area(
    "Enter Customer Review",
    height=150
)

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a review.")

    else:

        result = predict_sentiment(user_input)

        if "Positive" in result:
            st.success(f"Prediction: {result}")

        else:
            st.error(f"Prediction: {result}")

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")
st.caption("Built with Streamlit, PyTorch, Transformers, and NLP")