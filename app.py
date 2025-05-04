import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors import safe_open

# Load model function
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('model')
    model = BertForSequenceClassification.from_pretrained('model', use_safetensors=True)
    
    # Load weights from safetensors
    with safe_open("model/model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            model.state_dict()[key].copy_(tensor)
    
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prediction function
def predict_spam(text):
    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()[0]

# Streamlit UI
st.title("🚫 SMS Spam Detector")
st.write("Enter an SMS text to check if it's spam or not")

# Text input
email_text = st.text_area("SMS Text:", height=200)

# Prediction button
if st.button("Check Spam"):
    if email_text.strip() == "":
        st.warning("Please enter some text to analyze")
    else:
        # Get prediction
        probabilities = predict_spam(email_text)
        ham_prob = probabilities[0]
        spam_prob = probabilities[1]
        
        # Display results
        st.subheader("Results:")
        if spam_prob > ham_prob:
            st.error(f"🚨 **Spam Detected!** (Confidence: {spam_prob*100:.1f}%)")
        else:
            st.success(f"✅ **Ham (Not Spam)** (Confidence: {ham_prob*100:.1f}%)")
        
        