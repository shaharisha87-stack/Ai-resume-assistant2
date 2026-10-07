import os
import streamlit as st
import pypdf as pdf
import google.generativeai as genai
import json

# Define styling elements
st.set_page_config(page_title="AI ATS Resume Analyzer", page_icon="📄", layout="wide")

# Configure Google Gemini API Connection
# Pulling safely from environment variables for deployment security
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.sidebar.warning("⚠️ API Key not found in environment variables. Please provide it below to run locally.")
    api_key_input = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")
    if api_key_input:
        genai.configure(api_key=api_key_input)

# Helper function to extract text cleanly from uploaded PDF files
def extract_text_from_pdf(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Error parsing PDF file: {str(e)}")
        return None

# AI Analysis function executing the Gemini 1.5 Flash model
def analyze_resume_with_ai(resume_text, job_description):
    # Enforcing structural JSON formatting directly in the system prompt
    input_prompt = f"""
    You are an expert technical human resource manager and senior ATS (Applicant Tracking System) optimization specialist.
    Analyze the following resume against the provided job description. 
    
    Provide your evaluation strictly as a valid JSON object matching the following structure exactly. Do not include any markdown styling like ```json or backticks in your output text.
    
    Expected JSON Structure:
    {{
        "ats_score": 75,
        "profile_summary": "Short analytical summary matching the candidate against the role.",
        "missing_keywords": ["keyword1", "keyword2"],
        "critical_improvements": ["Improvement point 1", "Improvement point 2"],
        "formatting_feedback": "Observations on resume layout readability."
    }}

    Resume Text:
    {resume_text}

    Job Description:
    {job_description}
    """
    
    try:
        # Utilizing gemini-1.5-flash for rapid, cost-efficient analysis
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(input_prompt)
        
        # Clean potential AI markdown artifacts safely
        clean_text = response.text.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(clean_text)
        return parsed_json
    except json.JSONDecodeError:
        st.error("Failed to parse AI response into structural elements. Raw response printing below.")
        st.text(response.text)
        return None
    except Exception as e:
        st.error(f"AI API Execution Error: {str(e)}")
        return None

# Main Dashboard Layout
st.title("🚀 AI-Powered ATS Resume Analyzer")
st.markdown("Optimize your resume against target roles using advanced machine learning models.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Input Details")
    job_desc_input = st.text_area("Target Job Description:", placeholder="Paste the complete job description text here...", height=250)
    uploaded_resume = st.file_uploader("Upload your Resume:", type=["pdf"], help="Please provide your resume in PDF format.")

with col2:
    st.subheader("📊 Optimization Report")
    
    if st.button("Run ATS Evaluation", type="primary"):
        if not job_desc_input:
            st.warning("Please provide a target job description to match against.")
        elif not uploaded_resume:
            st.warning("Please upload a valid PDF resume file.")
        elif not (os.getenv("GEMINI_API_KEY") or 'api_key_input' in locals()):
            st.error("Missing Gemini API configuration. Please configure authorization rules.")
        else:
            with st.spinner("Analyzing text patterns and matching keywords..."):
                resume_parsed_text = extract_text_from_pdf(uploaded_resume)
                
                if resume_parsed_text:
                    report = analyze_resume_with_ai(resume_parsed_text, job_desc_input)
                    
                    if report:
                        score = report.get("ats_score", 0)
                        
                        # Dynamic color grouping based on metrics score
                        if score >= 80:
                            st.success(f"### ATS Score: **{score}%** (Strong Match)")
                        elif score >= 50:
                            st.warning(f"### ATS Score: **{score}%** (Needs Optimization)")
                        else:
                            st.error(f"### ATS Score: **{score}%** (Low Matching Rate)")
                        
                        st.markdown("#### 🎯 Profile Breakdown Summary")
                        st.write(report.get("profile_summary", "No summary generated."))
                        
                        st.markdown("#### 🔑 Missing Keywords & Core Skills")
                        missing_kw = report.get("missing_keywords", [])
                        if missing_kw:
                            st.write(", ".join([f"`{kw}`" for kw in missing_kw]))
                        else:
                            st.write("No major keywords missing!")
                            
                        st.markdown("#### 🛠️ Recommended Structural Changes")
                        for point in report.get("critical_improvements", []):
                            st.markdown(f"- {point}")
                            
                        st.markdown("#### 📐 Layout & Formatting Insights")
                        st.write(report.get("formatting_feedback", "Layout appears clean."))
