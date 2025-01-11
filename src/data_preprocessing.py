import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_data(file_path):
    """Load CSV dataset."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data."""
    data['label'] = data['label'].str.strip().str.lower()
    return data

def balance_dataset(df):
    """Balance all classes to have an equal number of samples."""
    label_counts = df['label'].value_counts()
    target_samples = label_counts.max()
    upsampled_data = [resample(df[df['label'] == label], 
                               replace=True, 
                               n_samples=target_samples, 
                               random_state=42) 
                      for label in df['label'].unique()]
    return pd.concat(upsampled_data).sample(frac=1, random_state=42)

def split_data(df, test_size=0.2):
    """
    Split the dataset ensuring enough samples for each class. 
    If stratification is not possible, it falls back to a random split.
    """
    min_class_count = df['label'].value_counts().min()
    
    # Check if stratification is possible
    if min_class_count < 2 or len(df['label'].unique()) > len(df) * test_size:
        print("⚠️ Warning: Insufficient data for stratification. Performing random split.")
        return train_test_split(df, test_size=test_size, random_state=42)
    else:
        # Perform stratified sampling when possible
        return train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)