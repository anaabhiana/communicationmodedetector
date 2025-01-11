import os

# Define the base directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure the path resolves correctly for the CSV file in the 'data' folder
TRAINING_FILE = os.path.join(BASE_DIR, "../data/balanced_communication_dataset.csv")


MODEL_NAME = "bert-base-uncased"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 5e-5

