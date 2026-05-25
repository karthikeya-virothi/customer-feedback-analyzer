# 💬 Customer Feedback Analyzer

An AI-powered Customer Feedback Analyzer built using **Python**, **NLP**, **Transformers**, and **Streamlit**.

This project analyzes customer reviews and predicts sentiment as:

* 😡 Negative
* 😐 Neutral
* 😊 Positive

The application supports:

* Single review sentiment analysis
* CSV bulk review analysis
* Interactive dashboard visualization
* Downloadable analysis reports

---

# 🚀 Features

## ✅ Single Review Analysis

Analyze one customer review instantly.

## ✅ CSV Dataset Analysis

Upload CSV files containing customer reviews for bulk sentiment prediction.

## ✅ Interactive Dashboard

Visualize sentiment distribution using:

* Bar Charts
* Pie Charts

## ✅ Confidence Scores

Displays model confidence for each sentiment class.

## ✅ Download Results

Export analyzed results as CSV.

## ✅ NLP + Transformers

Uses Hugging Face Transformer models for advanced sentiment analysis.

---

# 🛠️ Technologies Used

* Python
* Streamlit
* PyTorch
* Hugging Face Transformers
* Pandas
* Matplotlib
* NLP

---

# 📂 Project Structure

```bash
customer-feedback-analyzer/
│
├── app/
│   └── app.py
│
├── data/
│   └── Reviews.csv
│
├── models/
│   └── sentiment_model/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│
├── notebooks/
│   ├── Customer_Feedback_Analyzer.ipynb
│   └── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/karthikeya-virothi/customer-feedback-analyzer.git
```

---

## 2️⃣ Navigate to Project

```bash
cd customer-feedback-analyzer
```

---

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## 4️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
streamlit run app/app.py
```

---

# 📊 CSV Format

CSV file must contain a column named:

```text
text
```

Example:

| text                      |
| ------------------------- |
| This product is amazing   |
| Worst experience ever     |
| Package arrived yesterday |

---

# 🧠 Model Used

This project uses:

```text
cardiffnlp/twitter-roberta-base-sentiment
```

from Hugging Face Transformers.

---

# 📈 Dashboard Features

* Sentiment Distribution
* Positive vs Negative Ratio
* Pie Chart Visualization
* Dataset Insights
* Downloadable Results

---

# ☁️ Deployment

The project is deployed using:

* GitHub
* Streamlit Community Cloud

---

# 🔮 Future Enhancements

* Real-time API integration
* Multilingual sentiment analysis
* Aspect-based sentiment analysis
* Voice feedback analysis
* AI chatbot integration
* Advanced analytics dashboard
* Database integration
* User authentication system

---

# 👨‍💻 Author

## Karthikeya Virothi

AI/ML Enthusiast | NLP Developer | Python Developer

---

# 📜 License

This project is developed for educational and internship demonstration purposes.
