import torch
import numpy as np
import json
import re
import sys
from model import LSTMClassifier

# Hyperparameters (must match training)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
MAX_LEN = 200

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_pad(text, word_to_idx, max_len):
    tokens = [word_to_idx.get(word, 1) for word in text.split()]  # 1 is <UNK>
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))  # 0 is <PAD>
    else:
        tokens = tokens[:max_len]
    return tokens

def predict(resume_text):
    # Load mappings
    with open('word_to_idx.json', 'r') as f:
        word_to_idx = json.load(f)
        
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
        
    VOCAB_SIZE = len(word_to_idx)
    OUTPUT_DIM = len(label_mapping['label_to_idx'])
    idx_to_label = {int(k): v for k, v in label_mapping['idx_to_label'].items()}
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()
    
    # Preprocess input
    cleaned = clean_text(resume_text)
    tokenized = tokenize_and_pad(cleaned, word_to_idx, MAX_LEN)
    input_tensor = torch.LongTensor([tokenized]).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_class = prediction.argmax(1).item()
        probabilities = torch.softmax(prediction, dim=1)[0]
    
    predicted_label = idx_to_label[predicted_class]
    confidence = probabilities[predicted_class].item()
    
    print(f"\nPredicted Category: {predicted_label}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    print("\nTop 3 Predictions:")
    top_probs, top_indices = torch.topk(probabilities, min(3, OUTPUT_DIM))
    for prob, idx in zip(top_probs, top_indices):
        print(f"  {idx_to_label[idx.item()]}: {prob.item()*100:.2f}%")
    
    return predicted_label, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        resume_text = sys.argv[1]
    else:
        print("Enter resume text (or press Ctrl+C to exit):")
        resume_text = input()
    
    predict(resume_text)
