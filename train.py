import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import LSTMClassifier

# Hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
BATCH_SIZE = 64
N_EPOCHS = 10
LEARNING_RATE = 0.001

def plot_history(history):
    """Plot and save training history"""
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png")

def train():
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
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create DataLoaders
    train_data = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    val_data = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_valid_loss = float('inf')
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("Starting training...")
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            
            acc = (predictions.argmax(1) == labels).sum().item() / labels.size(0)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc
            
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                predictions = model(texts)
                loss = criterion(predictions, labels)
                
                acc = (predictions.argmax(1) == labels).sum().item() / labels.size(0)
                
                val_loss += loss.item()
                val_acc += acc
                
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f'Epoch: {epoch+1:02} | Train Loss: {avg_train_loss:.3f} | Train Acc: {avg_train_acc*100:.2f}% | Val Loss: {avg_val_loss:.3f} | Val Acc: {avg_val_acc*100:.2f}%')
        
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"Model saved to best_model.pt")
            
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    print("Training history saved to training_history.json")
    
    # Plot history
    plot_history(history)

if __name__ == "__main__":
    train()
