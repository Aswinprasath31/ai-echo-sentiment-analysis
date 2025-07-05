import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Clean function same as before
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

# Streamlit UI
st.set_page_config(page_title="AI Echo - Review Sentiment")
st.title("🧠 AI Echo: ChatGPT Review Sentiment Analyzer")

review_input = st.text_area("💬 Enter your ChatGPT review here:")

if st.button("🔍 Analyze Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review text.")
    else:
        cleaned = clean_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        st.success(f"🎯 Predicted Sentiment: **{prediction}**")

        if prediction == "Positive":
            st.balloons()
        elif prediction == "Negative":
            st.error("☹️ Seems like a bad experience!")
        else:
            st.info("😐 This feels neutral.")
