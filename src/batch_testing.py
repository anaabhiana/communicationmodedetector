import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import joblib
import pandas as pd
from src.config import MODEL_NAME

# Load model, tokenizer, and label encoder
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained("../models/bert_model")
label_encoder = joblib.load("../models/label_encoder.pkl")

# Generate label map from saved label encoder
label_map = {index: label for index, label in enumerate(label_encoder.classes_)}

# List of positive and negative sentiment keywords
positive_keywords = ["happy", "grateful", "content", "peaceful"]
negative_keywords = ["sucks", "terrible", "awful", "miserable", "bad"]

# Function to predict communication mode
def predict_mode(text):
    # Mixed sentiment handling
    if any(word in text.lower() for word in negative_keywords):
        if any(word in text.lower() for word in positive_keywords):
            return "Mixed Sentiment - Possibly Supportive or Critical"
        return "critical"
    
    # Tokenize and classify using the model
    tokens = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    outputs = model(**tokens)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_map.get(prediction, "Unknown Mode Detected")

# Batch testing function
def batch_test(csv_file):
    df = pd.read_csv(csv_file)
    df['Predicted Mode'] = df['message'].apply(predict_mode)
    return df

# Run batch test on a CSV file
if __name__ == "__main__":
    test_file = input("Enter the path to your test CSV file: ")
    result_df = batch_test(test_file)
    result_df.to_csv("../data/batch_test_results.csv", index=False)
    print("Batch classification completed. Results saved in '../data/batch_test_results.csv'")