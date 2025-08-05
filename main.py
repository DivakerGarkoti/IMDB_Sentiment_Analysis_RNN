import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load dataset and word index
max_features = 10000
(X_train, Y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the saved model
model = load_model("model.h5")

# Decode review from integers to words
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess text: lowercase, tokenize, encode, pad
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Predict sentiment from raw review text
def predict_sentiment(review):
    preprocessed_text = preprocess_text(review)
    prediction = model.predict(preprocessed_text)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return prediction[0][0], sentiment

# Streamlit app UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to classify it as **Positive** or **Negative**:")

# User input
user_input = st.text_area(" Movie Review")

# Classification logic
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        score, sentiment = predict_sentiment(user_input)
        st.markdown(f"### Sentiment: **{sentiment}**")
        st.markdown(f"### Prediction Score: `{score:.4f}`")
else:
    st.info("Awaiting your review input.")
