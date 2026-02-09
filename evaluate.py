import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from model import LSTMClassifier

# Hyperparameters (must match training)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

def evaluate():
    # Load data
    print("Loading preprocessed data...")
    X = np.load('X.npy')
    y = np.load('y.npy')
    
    with open('word_to_idx.json', 'r') as f:
        word_to_idx = json.load(f)
        
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
        
    VOCAB_SIZE = len(word_to_idx)
    OUTPUT_DIM = len(label_mapping['label_to_idx'])
    
    # Split data (same as training)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()
    
    # Evaluate
    X_val_tensor = torch.LongTensor(X_val).to(device)
    
    with torch.no_grad():
        predictions = model(X_val_tensor)
        predicted_labels = predictions.argmax(1).cpu().numpy()
    
    # Get label names
    idx_to_label = {int(k): v for k, v in label_mapping['idx_to_label'].items()}
    label_names = [idx_to_label[i] for i in range(OUTPUT_DIM)]
    
    print("\n=== Classification Report ===")
    print(classification_report(y_val, predicted_labels, target_names=label_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val, predicted_labels, label_names)
    
    # Calculate accuracy
    accuracy = (predicted_labels == y_val).sum() / len(y_val)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    evaluate()
