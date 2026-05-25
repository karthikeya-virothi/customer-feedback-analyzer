import streamlit as st
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Feedback Analyzer", layout="wide")

st.title("💬 Customer Feedback Analyzer")
st.markdown("### AI Sentiment Analysis Dashboard")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "sentiment_model")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return model, tokenizer, device

model, tokenizer, device = load_model()

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    labels = ["Negative 😡", "Neutral 😐", "Positive 😊"]
    pred = labels[int(probs.argmax())]

    return pred, probs


# -------------------------------
# SIDEBAR
# -------------------------------
mode = st.sidebar.radio("Select Mode", ["Single Review", "CSV Analysis"])

# -------------------------------
# SINGLE REVIEW
# -------------------------------
if mode == "Single Review":
    st.subheader("📝 Analyze Single Feedback")

    text = st.text_area("Enter feedback")

    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Enter text")
        else:
            sentiment, probs = predict(text)

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"Sentiment: {sentiment}")

            with col2:
                st.info("Confidence Scores")
                st.write({
                    "Negative": float(probs[0]),
                    "Neutral": float(probs[1]),
                    "Positive": float(probs[2])
                })

# -------------------------------
# CSV ANALYSIS
# -------------------------------
elif mode == "CSV Analysis":
    st.subheader("📂 Upload CSV")

    file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()

        if "text" not in df.columns:
            st.error("CSV must contain 'text' column")
        else:
            st.dataframe(df.head())

            if st.button("Analyze Dataset"):

                results = []

                with st.spinner("Analyzing dataset..."):
                    for txt in df["text"]:
                        sentiment, _ = predict(str(txt))
                        results.append(sentiment)

                df["Sentiment"] = results

                st.success("Analysis Complete 🚀")

                # -------------------------------
                # CHARTS
                # -------------------------------
                st.subheader("📊 Sentiment Distribution")

                sentiment_counts = df["Sentiment"].value_counts()

                col1, col2 = st.columns(2)

                with col1:
                    st.bar_chart(sentiment_counts)

                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
                    ax.axis("equal")
                    st.pyplot(fig)

                # -------------------------------
                # INSIGHTS
                # -------------------------------
                st.write("### 🧠 Insights")

                total = len(df)
                pos = sentiment_counts.get("Positive 😊", 0)
                neg = sentiment_counts.get("Negative 😡", 0)

                if total > 0:
                    pos_pct = round((pos / total) * 100, 2)
                    neg_pct = round((neg / total) * 100, 2)

                    st.write(f"✔ {pos_pct}% reviews are positive")
                    st.write(f"✔ {neg_pct}% reviews are negative")

                    if pos > neg:
                        st.success("Overall customer sentiment is GOOD 👍")
                    elif neg > pos:
                        st.error("Overall customer sentiment is NOT satisfied ⚠️")
                    else:
                        st.warning("Sentiment is BALANCED ⚖️")
                else:
                    st.warning("No data available for insights")

                # -------------------------------
                # TABLE + DOWNLOAD
                # -------------------------------
                st.subheader("📋 Results")

                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "📥 Download Results",
                    data=csv,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )