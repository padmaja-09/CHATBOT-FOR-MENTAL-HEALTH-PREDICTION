import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set Streamlit Page Configuration
st.set_page_config(page_title="Mental Health Chatbot", layout="centered", page_icon="💙")

# Custom CSS for full background color
st.markdown(
    """
    <style>
    body {
        background-color: #d1e7fd !important;
    }
    .stApp {
        background-color: #d1e7fd !important;
    }
    .stTextArea>label {
        font-size: 20px;
        font-weight: bold;
        color: #4a4a4a;
    }
    .stButton>button {
        background-color: #ff7eb3;
        color: white;
        font-size: 18px;
        padding: 8px 20px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #ff4f81;
    }
    .bot-response {
        font-size: 18px;
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        font-weight: bold;
        color: #1565c0;
    }
    .status-box {
        font-size: 20px;
        font-weight: bold;
        background-color: #ffd966;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
    else:
        text = ''
    return text

# Load dataset
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv('Combined Data.csv')

mental_health_data = load_data()
mental_health_data['cleaned_text'] = mental_health_data['statement'].apply(clean_text)

# Train Model
@st.cache_resource(show_spinner=False)
def train_model():
    import sys, os
    sys.stdout = open(os.devnull, 'w')  # Suppress training output

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(mental_health_data['cleaned_text'])
    y = mental_health_data['status'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    sys.stdout = sys.__stdout__  # Restore print functionality
    return model, vectorizer

model, vectorizer = train_model()

# Chatbot Response Function
def chatbot_response(user_input):
    cleaned_input = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)[0]

    responses = {
        "Anxiety": "💙 Anxiety can be overwhelming, but take a deep breath and know that you're not alone.",
        "Bipolar": "🌀 Managing bipolar disorder can be challenging. Seeking support from loved ones and professionals can help.",
        "Normal": "😊 It's great to hear that you're feeling okay! If you ever need to talk, I'm here to listen.",
        "Depression": "💔 I'm so sorry you're feeling this way. You're not alone, and there are people who care about you and want to help.",
        "Personality Disorder": "🌿 It must be tough dealing with this. Remember, you're strong, and with the right support, you can manage it.",
        "Stress": "🌸 Stress can be difficult to handle. Try to take breaks, practice deep breathing, and take care of yourself.",
        "Suicidal": "🚨 I'm really sorry you're feeling this way. Please reach out to someone you trust or a professional. You matter, and help is available."
    }

    predicted_status = prediction.strip().capitalize()
    response_text = responses.get(predicted_status, "🌻 I'm here to support you. You're not alone, and things can get better.")

    return predicted_status, response_text

# Streamlit App UI
st.title("💙 Mental Health Support Chatbot")
st.write("🌟 This AI-powered chatbot provides **supportive responses** based on your emotions.")

# Chatbot Section
st.subheader("💬 How are you feeling today?")
user_input = st.text_area("Type your message here...", height=100)

if st.button("Send 💌"):
    if user_input.strip():
        predicted_status, response = chatbot_response(user_input)
        st.markdown(f'<div class="status-box">🔹 Predicted Status: <b>{predicted_status}</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-response">🤖 {response}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message.")

# Footer
st.markdown("---")
st.markdown(
    "🔹 *This chatbot is for **informational purposes only** and should not be used as a substitute for professional mental health advice.*"
)
