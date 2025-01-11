# Multi-Label Communication Mode Detector
This project is designed for multi-label classification using a BERT model. It effectively fine-tunes a BERT model for text classification with attention to data preprocessing, class imbalance handling, and device optimization. The custom trainer ensures better handling of underrepresented classes using weighted loss calculation.



## 🧑‍💻 Setup Instructions:You need to run all of these command from project root directory 
1. **Create Virtual Environment:** 

If Python 3.11 is not installed, you can install it using Homebrew (for macOS):
 brew install python@3.11   

2.	Create a Virtual Environment with Python 3.11:
 python3.11 -m venv venv 
 source venv/bin/activate 


3.	Install PyTorch (CPU Version):
pip install torch torchvision torchaudio

4.	Verify Installation:
python -c "import torch; print(torch.__version__)"
 # it should show the version 

5.	Install Dependencies:
pip install -r requirements.txt

6.	Train the Model:
python src/train_model.py

7.	Run Inference:
python src/inference.py
