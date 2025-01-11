import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from config import MODEL_NAME

# ✅ Load the model, tokenizer, and label encoder
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained("../models/bert_model").to(device)
label_encoder = joblib.load("../models/label_encoder.pkl")

# ✅ Check if the dataset file exists before loading
import os

# Set an absolute path for the dataset file
dataset_path = os.path.abspath("./data/balanced_communication_dataset.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The dataset file was not found at {dataset_path}")
else:
    print(f"✅ Dataset found at: {dataset_path}")

# ✅ Load the test dataset
test_data = pd.read_csv(dataset_path)

# ✅ Function to predict communication mode
def predict_mode(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").to(device)
    outputs = model(**tokens)
    prediction_index = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([prediction_index])[0]
    return predicted_label

# ✅ Perform predictions and collect results
predictions = []
true_labels = []

for _, row in test_data.iterrows():
    true_labels.append(row["label"])
    predictions.append(predict_mode(row["message"]))

# ✅ Calculate and print performance metrics
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions)

print(f"Validation Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)