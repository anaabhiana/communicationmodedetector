import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import joblib
from config import MODEL_NAME

# Detect MPS or fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load model, tokenizer, and label encoder
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained("../models/bert_model").to(device)
label_encoder = joblib.load("../models/label_encoder.pkl")

def predict_mode(text):
    # Move inputs to the correct device
    tokens = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").to(device)
    outputs = model(**tokens)

    # Apply softmax to get probabilities for confidence scoring
    probabilities = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
    prediction_index = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([prediction_index])[0]
    confidence = probabilities[prediction_index]

    return f"{predicted_label} (Confidence: {confidence:.2f})"

if __name__ == "__main__":
    text_input = input("Enter a message to classify: ")
    prediction = predict_mode(text_input)
    print(f"Predicted Communication Mode: {prediction}")