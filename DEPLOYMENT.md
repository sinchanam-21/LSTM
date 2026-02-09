# 🚀 Deployment Guide: LSTM Resume Classifier

This guide provides several ways to deploy your Resume Classifier web application to the cloud.

## Recommended: Streamlit Community Cloud (Free & Easiest)

Streamlit Cloud is the fastest way to deploy, especially if your code is on GitHub.

1. **Push your code to GitHub**:
   - Ensure `app.py`, `model.py`, `best_model.pt`, `word_to_idx.json`, `label_mapping.json`, and `requirements.txt` are in your repository.
   - *Note: You don't need `Resume.csv` or `.npy` files for deployment.*
2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click "New app", select your repository, branch, and set the main file path to `app.py`.
4. Click **Deploy!**

---

## Alternative: Hugging Face Spaces (Free)

Great for machine learning apps.

1. Create a new "Space" on [Hugging Face](https://huggingface.co/spaces).
2. Select **Streamlit** as the SDK.
3. Upload your files or push via Git.
4. The app will automatically build and run.

---

## Advanced: Docker Deployment

Use this if you want to deploy to AWS, GCP, Azure, or DigitalOcean using containers.

### 1. Build the Docker Image
```bash
docker build -t resume-classifier .
```

### 2. Run Locally to Test
```bash
docker run -p 8501:8501 resume-classifier
```

### 3. Deploy to Cloud
- **AWS**: Push to ECR and run on ECS or App Runner.
- **GCP**: Push to Artifact Registry and run on Cloud Run.
- **Azure**: Use Azure Container Instances or App Service.

---

## 🛠️ Deployment Tips

- **Model Size**: Ensure `best_model.pt` is included. If it's very large (>100MB), you might need Git LFS on GitHub.
- **Dependencies**: Keep `requirements.txt` updated.
- **Secrets**: If you add API keys later, use Streamlit's secrets management instead of hardcoding them.
- **Resources**: LSTM models are lightweight enough to run on free tiers (like Streamlit Cloud or HF Spaces).

---
**Happy Deploying! 🌐**
