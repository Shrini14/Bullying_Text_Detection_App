import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load stopwords
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# Load TF-IDF vectorizer and model
vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True,
                             vocabulary=pickle.load(open("tfidfvectorizer.pkl", "rb")))
model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

# ---- UI Enhancements ---- #
st.set_page_config(page_title="Bullying Text Detection", page_icon="🚨", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .stTextArea textarea {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #007BFF !important;
            color: white !important;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App Header
st.title("🚨 Bullying Text Detection App")
st.markdown("### 💬 Detect whether a given text contains bullying content.")

# Input text from user
user_input = st.text_area("✍️ Enter text for classification:", height=150, placeholder="Type your text here...")

# Prediction Button
if st.button("🚀 Predict"):
    if user_input.strip():
        transformed_input = vectorizer.fit_transform([user_input])  # Use transform, not fit_transform
        prediction = model.predict(transformed_input)[0]

        # 🎯 Enhanced result display
        st.subheader("🎯 Prediction Result:")
        if prediction == 1:
            st.error("⚠️ This text **may contain bullying content**. Please be cautious!")
        else:
            st.success("✅ This text **does not contain bullying content**. It's safe to use!")
    else:
        st.warning("⚠️ Please enter some text before predicting.")

# Footer with styling
st.markdown("<hr style='border:1px solid #ddd;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#777;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
