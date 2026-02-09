# 🌐 Web Interface - Quick Start Guide

## ✅ Your Web Application is Running!

The Resume Classifier web application is now live and accessible through your browser.

## 🔗 Access URLs

**Local Access (from this computer):**
```
http://localhost:8501
```

**Network Access (from other devices on the same network):**
```
http://10.205.2.247:8501
```

## 🎯 How to Use

1. **Open your browser** and go to `http://localhost:8501`

2. **Enter Resume Text** in the text area, or click one of the sample buttons:
   - HR Sample
   - Accountant Sample
   - Engineer Sample

3. **Click "Classify Resume"** button

4. **View Results:**
   - Predicted Category
   - Confidence Score
   - Top 5 Predictions with probabilities

## 📱 Features

- ✨ **Clean, Modern UI** - Easy to use interface
- 📊 **Visual Confidence Indicators** - Color-coded confidence levels
- 🎯 **Top 5 Predictions** - See alternative categories with probabilities
- 📝 **Sample Resumes** - Quick test with pre-loaded examples
- 📋 **24 Categories** - Complete list in the sidebar
- ⚡ **Fast Predictions** - Instant results

## 🛑 How to Stop the Server

Press `Ctrl + C` in the terminal where Streamlit is running.

## 🔄 How to Restart

```bash
cd c:\LSTM
streamlit run app.py
```

## 📂 Project Files

- `app.py` - Streamlit web application
- `model.py` - LSTM model architecture
- `best_model.pt` - Trained model weights
- `word_to_idx.json` - Vocabulary mapping
- `label_mapping.json` - Category labels
- `requirements.txt` - Python dependencies

## 🌍 Sharing with Others

### Option 1: Local Network
Anyone on your local network can access using:
```
http://10.205.2.247:8501
```

### Option 2: Deploy to Cloud (for public access)
To make it accessible from anywhere on the internet, you can deploy to:
- **Streamlit Cloud** (Free) - https://streamlit.io/cloud
- **Heroku** - https://heroku.com
- **Google Cloud Platform**
- **AWS**

## 💡 Tips

- For best results, use complete resume text (not just keywords)
- The model performs better with resumes similar to the training data
- Confidence scores above 70% are generally very reliable
- Low confidence may indicate the resume doesn't fit standard categories

---

**Enjoy your AI-powered Resume Classifier! 🚀**
