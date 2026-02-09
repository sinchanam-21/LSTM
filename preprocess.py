import pandas as pd
import numpy as np
import re
import torch
import json
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def clean_text(text):
    # Remove HTML tags if any (though we use Resume_str)
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(csv_path, max_len=200, vocab_size=5000):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Use Resume_str for classification
    print("Cleaning text...")
    df['cleaned_resume'] = df['Resume_str'].apply(clean_text)
    
    # Encode labels
    print("Encoding labels...")
    categories = df['Category'].unique()
    label_to_idx = {cat: i for i, cat in enumerate(categories)}
    idx_to_label = {i: cat for i, cat in enumerate(categories)}
    df['label'] = df['Category'].map(label_to_idx)
    
    # Build vocabulary
    print("Building vocabulary...")
    all_text = ' '.join(df['cleaned_resume'].tolist())
    words = all_text.split()
    word_counts = Counter(words)
    # Keep most common words
    vocab = [word for word, count in word_counts.most_common(vocab_size - 2)] # -2 for <PAD> and <UNK>
    word_to_idx = {word: i + 2 for i, word in enumerate(vocab)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    
    def tokenize(text):
        return [word_to_idx.get(word, 1) for word in text.split()]
    
    print("Tokenizing and padding...")
    df['tokenized'] = df['cleaned_resume'].apply(tokenize)
    
    # Pad sequences
    def pad_sequence(seq):
        if len(seq) < max_len:
            return seq + [0] * (max_len - len(seq))
        else:
            return seq[:max_len]
            
    df['padded'] = df['tokenized'].apply(pad_sequence)
    
    # Prepare final data
    X = np.array(df['padded'].tolist())
    y = np.array(df['label'].tolist())
    
    # Save processed data and mappings
    print("Saving processed data...")
    np.save('X.npy', X)
    np.save('y.npy', y)
    
    with open('label_mapping.json', 'w') as f:
        json.dump({'label_to_idx': label_to_idx, 'idx_to_label': {str(k): v for k, v in idx_to_label.items()}}, f)
        
    with open('word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f)
        
    print("Preprocessing complete.")
    return X, y, label_to_idx, word_to_idx

if __name__ == "__main__":
    preprocess_data('Resume.csv')
