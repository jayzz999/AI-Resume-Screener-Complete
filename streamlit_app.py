import streamlit as st
import os
import tempfile
from parser_utils import parse_resume
from feature_extraction import FeatureExtractor
import traceback

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screener")
st.write("Upload a resume and provide a job description to analyze candidate fit")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=['pdf', 'docx'],
        help="Upload candidate's resume"
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

with col2:
    st.subheader("📝 Job Description")
    job_description = st.text_area(
        "Enter the job description",
        height=200,
        placeholder="Paste the job description here...",
        help="Provide detailed job requirements"
    )

if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
    if not uploaded_file:
        st.error("❌ Please upload a resume")
    elif not job_description.strip():
        st.error("❌ Please enter a job description")
    else:
        try:
            with st.spinner("🔄 Analyzing resume..."):
                # Save uploaded file to temporary path
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Parse resume using file path, not UploadedFile object
                    resume_result = parse_resume(tmp_file_path)
                    resume_text = resume_result["raw_text"] if isinstance(resume_result, dict) else resume_result
                    
                    if not resume_text or not resume_text.strip():
                        st.error("❌ Failed to extract text from resume")
                    else:
                        # Initialize feature extractor
                        extractor = FeatureExtractor()
                        
                        # Use the correct API: generate_explanation
                        explanation = extractor.generate_explanation(resume_text, job_description)
                        
                        # Extract score and details from explanation
                        overall_score = explanation['summary']['overall_score'] * 100
                        skills = explanation['details']
                        
                        # Store results
                        st.session_state.results = {
                            'score': overall_score,
                            'skills': skills,
                            'resume_text': resume_text[:500] + '...' if len(resume_text) > 500 else resume_text
                        }
                        st.session_state.analysis_complete = True
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                        
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            st.exception(e)
            st.code(traceback.format_exc())

# Display results if analysis is complete
if st.session_state.analysis_complete and st.session_state.results:
    st.divider()
    st.subheader("📊 Analysis Results")
    
    results = st.session_state.results
    
    # Display overall score
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            label="Overall Match Score",
            value=f"{results['score']:.1f}%",
            help="How well the candidate matches the job requirements"
        )
    
    # Display skills analysis
    st.subheader("🎯 Skills Analysis")
    
    if results['skills']:
        for skill, details in results['skills'].items():
            with st.expander(f"**{skill}**"):
                if isinstance(details, dict):
                    for key, value in details.items():
                        st.write(f"**{key}:** {value}")
                else:
                    st.write(details)
    else:
        st.info("No detailed skills analysis available")
    
    # Display resume preview
    with st.expander("📄 Resume Preview"):
        st.text(results['resume_text'])
    
    # Reset button
    if st.button("🔄 Analyze Another Resume", use_container_width=True):
        st.session_state.analysis_complete = False
        st.session_state.results = None
        st.rerun()

# Footer
st.divider()
st.caption("💡 Tip: Provide detailed job descriptions for more accurate analysis")
