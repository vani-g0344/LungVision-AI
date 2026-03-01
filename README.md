# LungVision-AI

- End-to-End AI System for Lung Cancer Detection from CT Scans

- LungVision-AI is a full-stack AI application that detects lung cancer from CT scans and explains its predictions in plain, human-readable language.

- The project combines deep learning, explainable AI, and generative AI to simulate a real-world medical AI pipeline—from image upload to interpretable results.

- This is not just a model. It’s a complete AI product.

# WHAT THE PROJECT DOES?

- User uploads a lung CT scan
- Deep learning model predicts Benign, Malignant, or Normal
- Grad-CAM highlights the region influencing the prediction
- A Large Language Model generates a clear explanation
- Results are shown via a web UI or Streamlit app

# MODEL PERFORMANCE

Metric	                |      Score
                        
- Training Accuracy	           100%
- Validation Accuracy          95.7%
- Test Accuracy	               95.8%

# SETUP

1. Clone the repo
2. pip install -r requirements.txt
3. python download_model.py   ← downloads model from Google Drive
4. cp .env.example .env and add your GROQ key
5. uvicorn backend/app:app --reload

Dataset -
- 1,097 CT scans
- 3 classes: Benign · Malignant · Normal
- Achieved using transfer learning and fine-tuning on a limited medical dataset, following standard medical AI research practices.

# TECHNICAL STACK

# Machine Learning / AI -
(i) PyTorch + TorchVision
(ii) ResNet50 (ImageNet pretrained, fine-tuned on CT scans)
(iii) Grad-CAM for visual explainability
(iv) LLaMA 3.1 (via Groq API) for natural language explanations

# Backend -
(i) FastAPI
(ii) Uvicorn
(iii) Python 3.11

# Frontend -
(i) HTML 
(ii) CSS 
(iii) JavaScript

# Training -
Google Colab (GPU)


# AUTHOR
Vani Gupta
Second-year | Computer Science Undergraduate
Aspiring Gen-AI Engineer · Full-Stack AI Developer


