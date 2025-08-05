# IMDB Movie Review Sentiment Analysis

This project performs sentiment analysis on IMDB movie reviews using a Recurrent Neural Network (RNN) built with Keras. The model classifies a review as either **Positive** or **Negative** based on its textual content. A simple and interactive web interface is created using Streamlit.

## Features

- Trained RNN model for sentiment classification
- Preprocessing of raw movie reviews into model-compatible format
- Streamlit-based user interface for real-time prediction
- Uses IMDB dataset from Keras

## Dataset

The IMDB dataset is automatically loaded via `tensorflow.keras.datasets.imdb`. It contains 50,000 movie reviews labeled as positive or negative.

- 25,000 training samples
- 25,000 test samples
- Vocabulary size limited to top 10,000 frequent words

## Model Overview

- Embedding layer for word representation
- Recurrent layer (e.g., LSTM or SimpleRNN)
- Dense layer with sigmoid activation for binary classification
- Trained using binary crossentropy loss and Adam optimizer

## Files

- `main.py` – Streamlit web app for entering and classifying movie reviews
- `model.h5` – Trained Keras RNN model file
- `README.md` – Project documentation
