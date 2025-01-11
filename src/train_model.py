import torch
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
import pandas as pd
from data_preprocessing import load_data, preprocess_data, balance_dataset, split_data
from config import MODEL_NAME, EPOCHS, BATCH_SIZE, LEARNING_RATE, TRAINING_FILE
import joblib
from collections import Counter
from transformers import BertTokenizerFast

# Detect MPS or fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load and prepare the dataset
df = load_data(TRAINING_FILE)
df = preprocess_data(df)
df = balance_dataset(df)
train_data, val_data = split_data(df)

# Tokenize and encode labels
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
train_tokens = tokenizer(list(train_data['message']), truncation=True, padding=True, return_tensors="pt")
val_tokens = tokenizer(list(val_data['message']), truncation=True, padding=True, return_tensors="pt")

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_data['label'])
val_labels_encoded = label_encoder.transform(val_data['label'])

# Save the label encoder for inference
joblib.dump(label_encoder, "../models/label_encoder.pkl")

# Handle class imbalance using class weights
label_counts = Counter(train_labels_encoded)
total_samples = sum(label_counts.values())
class_weights_tensor = torch.tensor([total_samples / count for count in label_counts.values()]).to(device)

# Prepare datasets for Hugging Face Trainer
train_dataset = Dataset.from_dict({
    "input_ids": train_tokens["input_ids"], 
    "attention_mask": train_tokens["attention_mask"], 
    "labels": torch.tensor(train_labels_encoded).to(device)
})
val_dataset = Dataset.from_dict({
    "input_ids": val_tokens["input_ids"], 
    "attention_mask": val_tokens["attention_mask"], 
    "labels": torch.tensor(val_labels_encoded).to(device)
})

# Load model and move it to the appropriate device
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))
model.to(device)

# Define Training Arguments with consistent save and eval strategies
training_args = TrainingArguments(
    output_dir="../models/bert_model/",
    eval_strategy="epoch",  
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True
)

# Custom Trainer Class for Weighted Loss Handling with MPS Fix
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Applying weighted loss for class imbalance
        loss_function = CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_function(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Initialize the custom trainer
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model using the corrected Trainer
trainer.train()
model.save_pretrained("../models/bert_model/")
print(f"Model trained and saved successfully with {len(label_encoder.classes_)} classes.")