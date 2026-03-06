import streamlit as st
import pickle

# load model and vectorizer
model = pickle.load(open("emotion_TfidfLog_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

emotion_labels = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}

st.title("Emotion Detection using NLP")

user_input = st.text_area("Enter text")

if st.button("Predict Emotion"):

    text_vector = vectorizer.transform([user_input])

    prediction = model.predict(text_vector)
    emotion = emotion_labels[prediction[0]]

    st.write("Predicted Emotion:", emotion)