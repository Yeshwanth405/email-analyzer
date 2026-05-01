import streamlit as st
import pickle
import re

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# FUNCTIONS
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def predict_email(text):
    cleaned = clean_text(text)

    urgent_keywords = ['asap', 'urgent', 'immediately', 'now']
    if any(word in cleaned for word in urgent_keywords):
        return "Urgent", 1.0

    if cleaned.strip() == "":
        return "Informational", 0.0

    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    try:
        confidence = model.decision_function(vector).max()
    except:
        confidence = 0.5

    return prediction, round(float(confidence), 2)

def generate_reply(intent):
    intent = intent.lower()

    if "urgent" in intent:
        return "We have received your urgent request and will respond shortly."
    elif "action" in intent:
        return "Thank you for your request. We will take action soon."
    else:
        return "Thank you for the information."

# -------------------------------
# 🎨 CSS DESIGN (PASTE HERE)
# -------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #2c3e50;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-top: 20px;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
}

.result {
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    padding: 10px;
    border-radius: 8px;
}
.urgent { background-color: #ff4d4d; color: white; }
.action { background-color: #ffa500; color: white; }
.info { background-color: #3498db; color: white; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# UI STARTS HERE
# -------------------------------

# Title
st.markdown('<div class="main-title">📧 AI Email Analyzer</div>', unsafe_allow_html=True)

# Input card
st.markdown('<div class="card">', unsafe_allow_html=True)

user_input = st.text_area("Enter your email text:")
analyze = st.button("Analyze Email")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# OUTPUT
# -------------------------------
if analyze:
    if user_input.strip() == "":
        st.warning("Please enter email text")
    else:
        label, confidence = predict_email(user_input)

        # Color logic
        if label == "Urgent":
            css_class = "urgent"
        elif label == "Action":
            css_class = "action"
        else:
            css_class = "info"

        # Prediction box
        st.markdown(f'<div class="result {css_class}">Prediction: {label}</div>', unsafe_allow_html=True)

        # Confidence
        st.write("Confidence Score:")
        st.progress(min(abs(confidence)/5, 1.0))
        st.write(confidence)

        # Reply
        reply = generate_reply(label)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("💬 Suggested Reply:")
        st.success(reply)
        st.markdown('</div>', unsafe_allow_html=True)