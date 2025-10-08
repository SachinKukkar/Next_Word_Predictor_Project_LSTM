import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------
# CUSTOM CSS FOR STYLING
# ----------------------------
st.markdown("""
    <style>
        /* Gradient background */
        body {
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            color: #f5f6fa;
            font-family: 'Poppins', sans-serif;
        }

        /* Streamlit app title */
        .title {
            text-align: center;
            font-size: 2.4em;
            font-weight: 700;
            color: #f9ca24;
            margin-bottom: 10px;
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #dcdde1;
            font-size: 1.1em;
            margin-bottom: 40px;
        }

        /* Input box styling */
        .stTextInput>div>div>input {
            background-color: #2f3640;
            color: white;
            border-radius: 10px;
            border: 1px solid #718093;
            font-size: 1.1em;
            padding: 10px;
        }

        /* Predict button styling */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00a8ff, #9c88ff);
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            transition: 0.3s;
        }

        div.stButton > button:first-child:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #9c88ff, #00a8ff);
        }

        /* Predicted word text */
        .result {
            text-align: center;
            font-size: 1.5em;
            margin-top: 30px;
            color: #4cd137;
            font-weight: 600;
        }

    </style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL & TOKENIZER
# ----------------------------
@st.cache_resource
def load_lstm_model():
    model = load_model('next_word_lstm.h5')
    return model

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

model = load_lstm_model()
tokenizer = load_tokenizer()

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# ----------------------------
# UI ELEMENTS
# ----------------------------
st.markdown("<div class='title'>🧠 Next Word Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by LSTM Neural Network trained on Shakespeare's Hamlet</div>", unsafe_allow_html=True)

input_text = st.text_input("Enter the sequence of words:", "To be or not to")
if st.button("🔮 Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.markdown(f"<div class='result'>✨ Predicted Next Word: <b>{next_word}</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result' style='color:#e84118;'>⚠️ Could not predict next word. Try a different input!</div>", unsafe_allow_html=True)
