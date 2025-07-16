🎯 Fake Job Posting Detector – NLP & Machine Learning Project

Built a project to detect fake job postings using a combination of natural language processing (NLP), statistical features, and classic ML models — deployed and ready for real-world usage.

🔍 Project Summary
The model takes raw job posting data and analyzes it for suspicious language patterns and structural features to determine the likelihood of it being fake. This solves a real-world problem for job seekers who fall prey to fraudulent ads.

🔧 Pipeline Overview
Data Source: Kaggle “Fake Job Postings” dataset (~18k rows)

Text Preprocessing:

Lowercasing

Removal of HTML, emails, URLs, digits, special characters

TF-IDF vectorization (up to 10k features)

Feature Engineering:

Text length, word count

Binary flags for presence of salary, benefits, requirements, etc.

Pattern detection for phrases like "easy money", "work from home"

Models Trained:

Logistic Regression

Random Forest (best performing with ~94% accuracy)

📦 Deployment Readiness
Saved models: joblib for TF-IDF, model, and metadata

Includes test prediction function to simulate inference

Ready to be hooked into Flask API or Gradio frontend

Hugging Face Spaces-ready

🧠 What I Learned
How to handle unstructured job data with real-world variability

Value of combining text + structured features

Performance tradeoffs between ML models and interpretation

How to create reusable pipelines with joblib

If you're hiring interns, mentoring, or just love NLP problems — happy to share notes, collab, or take feedback!

#NLP #MachineLearning #FakeJobDetection #InternshipReady #Python #AIProjects #DataScience #Gradio #HuggingFace #JobScams
# Fake_Job_Detector
