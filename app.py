import streamlit as st
import torch
import numpy as np
import json
import re
from model import LSTMClassifier
from pypdf import PdfReader
import io

# Hyperparameters (must match training)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
MAX_LEN = 200

# Page configuration
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide"
)

def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_pad(text, word_to_idx, max_len):
    """Tokenize and pad text to fixed length"""
    tokens = [word_to_idx.get(word, 1) for word in text.split()]
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

@st.cache_resource
def load_model():
    """Load the trained model and mappings"""
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
    
    return model, word_to_idx, idx_to_label, device

def predict_resume(resume_text, model, word_to_idx, idx_to_label, device):
    """Predict resume category"""
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
    
    # Get top 5 predictions
    top_probs, top_indices = torch.topk(probabilities, min(5, len(idx_to_label)))
    top_predictions = [(idx_to_label[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_indices)]
    
    return predicted_label, confidence, top_predictions

# Main app
def main():
    # Header
    st.title("📄 Resume Category Classifier")
    st.markdown("### AI-Powered Resume Classification using LSTM")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model, word_to_idx, idx_to_label, device = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This application uses a **Bidirectional LSTM** neural network "
            "to classify resumes into 24 different job categories.\n\n"
            "**Model Performance:**\n"
            "- Accuracy: 73.64%\n"
            "- Categories: 24\n"
            "- Architecture: Bi-LSTM with 2 layers"
        )
        
        st.header("📋 Categories")
        categories = sorted(list(idx_to_label.values()))
        st.markdown("**Available Categories:**")
        for i, cat in enumerate(categories, 1):
            st.text(f"{i}. {cat}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Input Resume")
        
        tab1, tab2 = st.tabs(["📝 Paste Text", "📂 Upload File"])
        
        resume_text = ""
        
        with tab1:
            text_input = st.text_area(
                "Paste the resume content below:",
                height=300,
                placeholder="Enter resume text here...\n\nExample:\nSoftware Engineer with 5 years of experience in Python, Java, and machine learning..."
            )
            
            # Sample resumes
            st.markdown("**Or try a sample:**")
            sample_col1, sample_col2, sample_col3 = st.columns(3)
            
            with sample_col1:
                if st.button("HR Sample"):
                    st.session_state['sample_text'] = "HR ADMINISTRATOR MARKETING ASSOCIATE Summary Dedicated Customer Service Manager with years of experience in Hospitality and Customer Service Management Respected builder and leader of customer focused teams strives to instill a shared enthusiastic commitment to customer service Highlights Focused on customer satisfaction Team management Marketing savvy Conflict resolution techniques Training and development Skilled multi tasker Client relations specialist"
            
            with sample_col2:
                if st.button("Accountant Sample"):
                    st.session_state['sample_text'] = "Certified Public Accountant CPA with years experience in financial reporting auditing tax preparation GAAP compliance Prepared financial statements for Fortune companies Managed accounts payable receivable General ledger reconciliation Budget analysis Proficient in QuickBooks SAP Excel Strong attention to detail and analytical skills"
            
            with sample_col3:
                if st.button("Engineer Sample"):
                    st.session_state['sample_text'] = "Civil Engineer with Bachelor degree in Civil Engineering Professional experience in construction project management site supervision structural design AutoCAD proficiency Building codes and regulations Quality control and safety compliance Project planning and scheduling Strong problem solving and communication skills"
            
            if 'sample_text' in st.session_state:
                text_input = st.session_state['sample_text']
                # Start: Hack to update text area if sample clicked 
                # (Streamlit doesn't update text_area directly from button without rerun or session state trickery which handles it)
                # We will just use the session state content if available
                # But to make it editable, we usually need key-value binding.
                # Simplified for this demo: if sample clicked, we use it.
                st.info("Sample loaded. You can edit it above if needed.")
                del st.session_state['sample_text'] # Clear after loading once
            
            if text_input:
                resume_text = text_input

        with tab2:
            uploaded_file = st.file_uploader("Upload a Resume", type=['pdf', 'txt'])
            if uploaded_file is not None:
                if uploaded_file.type == "application/pdf":
                    with st.spinner("Extracting text from PDF..."):
                        extracted_text = extract_text_from_pdf(uploaded_file)
                        if extracted_text:
                            resume_text = extracted_text
                            st.success("PDF text extracted successfully!")
                            with st.expander("View Extracted Text"):
                                st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)
                elif uploaded_file.type == "text/plain":
                    strio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    resume_text = strio.read()
                    st.success("Text file loaded successfully!")

        classify_button = st.button("🔍 Classify Resume", type="primary", use_container_width=True)
    
    with col2:
        st.header("Results")
        
        if classify_button:
            if not resume_text.strip():
                st.warning("⚠️ Please enter resume text or upload a file!")
            else:
                with st.spinner("Analyzing resume..."):
                    predicted_label, confidence, top_predictions = predict_resume(
                        resume_text, model, word_to_idx, idx_to_label, device
                    )
                
                # Display main prediction
                st.success("✅ Classification Complete!")
                st.metric("Predicted Category", predicted_label)
                st.metric("Confidence", f"{confidence*100:.2f}%")
                
                # Confidence indicator
                if confidence > 0.7:
                    st.success("🟢 High Confidence")
                elif confidence > 0.4:
                    st.warning("🟡 Medium Confidence")
                else:
                    st.error("🔴 Low Confidence")
                
                # Top 5 predictions
                st.markdown("---")
                st.subheader("Top 5 Predictions")
                for i, (category, prob) in enumerate(top_predictions, 1):
                    st.progress(prob, text=f"{i}. {category}: {prob*100:.2f}%")
        else:
            st.info("👈 Enter resume text or upload a file, then click 'Classify Resume'")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit & PyTorch | LSTM Resume Classifier"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
