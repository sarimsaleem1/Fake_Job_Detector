import gradio as gr
import pandas as pd
import joblib
import re
import json
import numpy as np
from scipy.sparse import hstack
import warnings
import os
import PyPDF2
from PIL import Image
import pytesseract
import io
warnings.filterwarnings('ignore')

# Check if model files exist
def check_model_files():
    required_files = [
        'fake_job_detector_model.pkl',
        'tfidf_vectorizer.pkl', 
        'feature_columns.pkl',
        'model_metadata.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

# Load the saved model and components
missing_files = check_model_files()
if missing_files:
    print(f"⚠️ Missing files: {missing_files}")
    print("Please upload these files to your Hugging Face Space:")
    for file in missing_files:
        print(f"  - {file}")
    model = None
    tfidf = None
    feature_columns = []
    metadata = {}
else:
    try:
        model = joblib.load('fake_job_detector_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        # Load metadata
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {metadata.get('model_type', 'Unknown')}")
        print(f"Training accuracy: {metadata.get('accuracy', 0):.4f}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        tfidf = None
        feature_columns = []
        metadata = {}

# PDF text extraction function
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        if pdf_file is None:
            return ""
        
        # Read PDF file
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # Extract text from all pages
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        return text.strip()
    
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Image text extraction function (OCR)
def extract_text_from_image(image_file):
    """Extract text from uploaded image using OCR"""
    try:
        if image_file is None:
            return ""
        
        # Open image
        image = Image.open(image_file)
        
        # Use OCR to extract text
        text = pytesseract.image_to_string(image)
        
        return text.strip()
    
    except Exception as e:
        return f"Error reading image: {str(e)}"

# Preprocessing function (same as training)
def preprocess_text(text):
    """Comprehensive text preprocessing function"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

# Suspicious patterns (same as training)
suspicious_patterns = [
    r'work from home',
    r'easy money',
    r'no experience required',
    r'flexible hours',
    r'immediate start',
    r'earn \$\d+',
    r'guaranteed income',
    r'part time',
    r'full time',
    r'make money',
    r'weekly pay'
]

employment_types = ['full-time', 'part-time', 'contract', 'temporary', 'internship']

def predict_job_posting(job_title, company_profile, job_description, requirements, benefits, location, salary_range, employment_type, required_experience, required_education, has_company_logo, has_questions):
    """Predict if a job posting is fake or real"""
    
    if model is None:
        return "❌ Error: Model not loaded. Please ensure all required files are uploaded.", "N/A", "Required files: fake_job_detector_model.pkl, tfidf_vectorizer.pkl, feature_columns.pkl, model_metadata.json"
    
    try:
        # Create test dataframe
        test_data = {
            'title': job_title or "",
            'company_profile': company_profile or "",
            'description': job_description or "",
            'requirements': requirements or "",
            'benefits': benefits or "",
            'location': location or "",
            'department': "",
            'salary_range': salary_range or "",
            'employment_type': employment_type or "",
            'required_experience': required_experience or "",
            'required_education': required_education or "",
            'has_company_logo': 1 if has_company_logo else 0,
            'has_questions': 1 if has_questions else 0
        }
        
        test_df = pd.DataFrame([test_data])
        
        # Create combined text
        test_df['combined_text'] = (test_df['title'] + ' ' +
                                  test_df['company_profile'] + ' ' +
                                  test_df['description'] + ' ' +
                                  test_df['requirements'] + ' ' +
                                  test_df['benefits'] + ' ' +
                                  test_df['location'] + ' ' +
                                  test_df['department'] + ' ' +
                                  test_df['salary_range'])
        
        # Preprocess text
        test_df['processed_text'] = test_df['combined_text'].apply(preprocess_text)
        
        # Create features
        test_df['text_length'] = test_df['processed_text'].apply(len)
        test_df['word_count'] = test_df['processed_text'].apply(lambda x: len(x.split()))
        test_df['has_company_logo'] = test_df['has_company_logo'].astype(int)
        test_df['has_questions'] = test_df['has_questions'].astype(int)
        test_df['has_salary_range'] = test_df['salary_range'].apply(lambda x: 1 if x else 0)
        test_df['has_requirements'] = test_df['requirements'].apply(lambda x: 1 if x else 0)
        test_df['has_benefits'] = test_df['benefits'].apply(lambda x: 1 if x else 0)
        
        # Employment type features
        for emp_type in employment_types:
            test_df[f'is_{emp_type}'] = test_df['employment_type'].str.contains(emp_type, case=False, na=False).astype(int)
        
        # Experience and education features
        test_df['requires_experience'] = test_df['required_experience'].apply(lambda x: 1 if x else 0)
        test_df['entry_level'] = test_df['required_experience'].str.contains('entry', case=False, na=False).astype(int)
        test_df['senior_level'] = test_df['required_experience'].str.contains('senior|executive', case=False, na=False).astype(int)
        test_df['requires_education'] = test_df['required_education'].apply(lambda x: 1 if x else 0)
        test_df['requires_degree'] = test_df['required_education'].str.contains('degree|bachelor|master|phd', case=False, na=False).astype(int)
        
        # Suspicious patterns
        for i, pattern in enumerate(suspicious_patterns):
            test_df[f'suspicious_{i}'] = test_df['processed_text'].str.contains(pattern, case=False, na=False).astype(int)
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in test_df.columns:
                test_df[col] = 0
        
        # Select features
        test_features = test_df[feature_columns]
        
        # Transform text
        text_tfidf = tfidf.transform(test_df['processed_text'])
        
        # Combine features
        combined_features = hstack([text_tfidf, test_features.values])
        
        # Make prediction
        prediction = model.predict(combined_features)[0]
        probability = model.predict_proba(combined_features)[0]
        
        # Format result
        result = "🚨 FAKE JOB POSTING" if prediction == 1 else "✅ LEGITIMATE JOB POSTING"
        confidence = f"{probability[prediction]:.2%}"
        
        # Additional details
        details = f"Real Job Probability: {probability[0]:.2%}\nFake Job Probability: {probability[1]:.2%}"
        
        return result, confidence, details
        
    except Exception as e:
        return f"Error during prediction: {str(e)}", "N/A", "Please check your inputs and try again."

def process_file_upload(pdf_file, image_file):
    """Process uploaded PDF or image files"""
    extracted_text = ""
    
    if pdf_file is not None:
        extracted_text += extract_text_from_pdf(pdf_file) + "\n\n"
    
    if image_file is not None:
        extracted_text += extract_text_from_image(image_file)
    
    if extracted_text.strip():
        return extracted_text.strip()
    else:
        return "No text extracted from uploaded files."

def analyze_uploaded_file(pdf_file, image_file):
    """Analyze job posting from uploaded file"""
    extracted_text = process_file_upload(pdf_file, image_file)
    
    if "Error" in extracted_text or "No text extracted" in extracted_text:
        return extracted_text, "", "", ""
    
    # Use extracted text as job description
    result, confidence, details = predict_job_posting(
        job_title="",
        company_profile="",
        job_description=extracted_text,
        requirements="",
        benefits="",
        location="",
        salary_range="",
        employment_type="",
        required_experience="",
        required_education="",
        has_company_logo=False,
        has_questions=False
    )
    
    return extracted_text, result, confidence, details

# Create Gradio interface
with gr.Blocks(title="Fake Job Posting Detector", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🕵️ Fake Job Posting Detector")
    gr.Markdown("### Analyze job postings by entering details manually or uploading PDF/image files")
    
    # Check if model is loaded
    if model is None:
        gr.Markdown("## ⚠️ Model Not Loaded")
        gr.Markdown("Please upload these required files to your Hugging Face Space:")
        missing_files = check_model_files()
        for file in missing_files:
            gr.Markdown(f"- **{file}**")
        gr.Markdown("After uploading, refresh the page.")
    
    with gr.Tabs():
        # Tab 1: Manual Input
        with gr.TabItem("📝 Manual Input"):
            with gr.Row():
                with gr.Column(scale=2):
                    job_title = gr.Textbox(label="Job Title", placeholder="e.g., Software Engineer")
                    company_profile = gr.Textbox(label="Company Profile", placeholder="Brief description of the company", lines=2)
                    job_description = gr.Textbox(label="Job Description", placeholder="Detailed job description", lines=4)
                    requirements = gr.Textbox(label="Requirements", placeholder="Job requirements and qualifications", lines=3)
                    benefits = gr.Textbox(label="Benefits", placeholder="Benefits offered", lines=2)
                    
                with gr.Column(scale=1):
                    location = gr.Textbox(label="Location", placeholder="e.g., New York, NY")
                    salary_range = gr.Textbox(label="Salary Range", placeholder="e.g., $50,000 - $70,000")
                    employment_type = gr.Dropdown(
                        choices=["", "Full-time", "Part-time", "Contract", "Temporary", "Internship"],
                        label="Employment Type"
                    )
                    required_experience = gr.Dropdown(
                        choices=["", "Entry level", "Mid-level", "Senior level", "Executive"],
                        label="Required Experience"
                    )
                    required_education = gr.Dropdown(
                        choices=["", "High school", "Bachelor's degree", "Master's degree", "PhD"],
                        label="Required Education"
                    )
                    has_company_logo = gr.Checkbox(label="Has Company Logo")
                    has_questions = gr.Checkbox(label="Has Screening Questions")
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Check Job Posting", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    result_output = gr.Textbox(label="Result", lines=1, interactive=False)
                    confidence_output = gr.Textbox(label="Confidence", lines=1, interactive=False)
                    details_output = gr.Textbox(label="Detailed Probabilities", lines=3, interactive=False)
        
        # Tab 2: File Upload
        with gr.TabItem("📄 Upload File"):
            gr.Markdown("### Upload a PDF or image file containing a job posting")
            
            with gr.Row():
                with gr.Column():
                    pdf_upload = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    image_upload = gr.File(
                        label="Upload Image (JPG, PNG)",
                        file_types=[".jpg", ".jpeg", ".png"],
                        type="filepath"
                    )
                    
                    analyze_file_btn = gr.Button("📊 Analyze Uploaded File", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    extracted_text_output = gr.Textbox(label="Extracted Text", lines=8, interactive=False)
                    file_result_output = gr.Textbox(label="Analysis Result", lines=1, interactive=False)
                    file_confidence_output = gr.Textbox(label="Confidence", lines=1, interactive=False)
                    file_details_output = gr.Textbox(label="Detailed Probabilities", lines=3, interactive=False)
    
    # Event handlers
    submit_btn.click(
        predict_job_posting,
        inputs=[job_title, company_profile, job_description, requirements, benefits, 
                location, salary_range, employment_type, required_experience, 
                required_education, has_company_logo, has_questions],
        outputs=[result_output, confidence_output, details_output]
    )
    
    clear_btn.click(
        lambda: [""] * 11 + [False] * 2 + [""] * 3,
        outputs=[job_title, company_profile, job_description, requirements, benefits,
                location, salary_range, employment_type, required_experience,
                required_education, has_company_logo, has_questions,
                result_output, confidence_output, details_output]
    )
    
    analyze_file_btn.click(
        analyze_uploaded_file,
        inputs=[pdf_upload, image_upload],
        outputs=[extracted_text_output, file_result_output, file_confidence_output, file_details_output]
    )
    
    # Examples
    gr.Markdown("### 📝 Example Job Postings to Test")
    
    examples = [
        [
            "Work from Home Data Entry",
            "Fast growing company",
            "Easy work from home! No experience needed! Earn $5000 weekly! Just send us your personal info and start immediately! Contact us now for this amazing opportunity!",
            "No experience required",
            "High pay, flexible hours",
            "Work from home",
            "$5000/week",
            "Part-time",
            "Entry level",
            "",
            False,
            False
        ],
        [
            "Software Engineer III",
            "Leading technology company with 10+ years in the industry",
            "We are seeking a skilled Software Engineer to join our development team. You will work on cutting-edge projects using modern technologies and collaborate with cross-functional teams.",
            "5+ years experience in software development, proficiency in Python/Java, knowledge of cloud platforms",
            "Health insurance, 401k, flexible PTO",
            "San Francisco, CA",
            "$120,000 - $150,000",
            "Full-time",
            "Senior level",
            "Bachelor's degree",
            True,
            True
        ]
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[job_title, company_profile, job_description, requirements, benefits,
                location, salary_range, employment_type, required_experience,
                required_education, has_company_logo, has_questions],
        outputs=[result_output, confidence_output, details_output],
        fn=predict_job_posting,
        cache_examples=False
    )
    
    gr.Markdown("### ⚠️ Disclaimer")
    gr.Markdown("This tool is for educational purposes only. Always verify job postings through official channels and be cautious when sharing personal information.")

# Launch the app
if __name__ == "__main__":
    demo.launch()