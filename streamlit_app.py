import streamlit as st
import os
import tempfile
from feature_extraction import FeatureExtractor
from parser_utils import extract_text_from_pdf, extract_text_from_docx

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 AI Resume Screener")
st.markdown("""
    Upload a resume and provide a job description to automatically screen and score the candidate.
    The system evaluates skills, experience, and education to provide a comprehensive match score.
""")
st.divider()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT"
    )

with col2:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Enter the job description",
        height=200,
        placeholder="Paste the job description here..."
    )

# Submit button
submit_button = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

if submit_button:
    if not uploaded_file:
        st.error("⚠️ Please upload a resume file.")
    elif not job_description:
        st.error("⚠️ Please provide a job description.")
    else:
        with st.spinner("Analyzing resume..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract text based on file type
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                
                if file_extension == '.pdf':
                    resume_text = extract_text_from_pdf(tmp_file_path)
                elif file_extension == '.docx':
                    resume_text = extract_text_from_docx(tmp_file_path)
                else:  # .txt
                    with open(tmp_file_path, 'r', encoding='utf-8') as f:
                        resume_text = f.read()
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                if resume_text:
                    # Extract features using FeatureExtractor class
                    extractor = FeatureExtractor()
                    features = extractor.extract_all_features(resume_text, job_description)
                    overall_score = extractor.calculate_overall_score(features)
                    explanation = extractor.generate_explanation(features, overall_score)
                    
                    # Update result dictionary to use explanation data
                    result = {
                        'overall_score': overall_score,
                        'matched_skills': explanation.get('matched_skills', []),
                        'missing_skills': explanation.get('missing_skills', []),
                        'additional_skills': explanation.get('additional_skills', []),
                        'recommendation': explanation.get('recommendation', 'No recommendation available')
                    }
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Overall score display
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric(
                            label="Overall Match Score",
                            value=f"{overall_score}%",
                            delta=None
                        )
                    
                    st.divider()
                    
                    # Skills analysis
                    st.subheader("🎯 Skills Analysis")
                    
                    skill_col1, skill_col2, skill_col3 = st.columns(3)
                    
                    with skill_col1:
                        matched_skills = result.get('matched_skills', [])
                        if matched_skills:
                            st.success(f"**✓ Matched Skills ({len(matched_skills)})**")
                            for skill in matched_skills:
                                st.write(f"• {skill}")
                        else:
                            st.success("**✓ Matched Skills (0)**")
                            st.write("No matched skills found.")
                    
                    with skill_col2:
                        missing_skills = result.get('missing_skills', [])
                        if missing_skills:
                            st.warning(f"**⚠ Missing Skills ({len(missing_skills)})**")
                            for skill in missing_skills:
                                st.write(f"• {skill}")
                        else:
                            st.warning("**⚠ Missing Skills (0)**")
                            st.write("No missing skills.")
                    
                    with skill_col3:
                        additional_skills = result.get('additional_skills', [])
                        if additional_skills:
                            st.info(f"**ℹ Additional Skills ({len(additional_skills)})**")
                            for skill in additional_skills:
                                st.write(f"• {skill}")
                        else:
                            st.info("**ℹ Additional Skills (0)**")
                            st.write("No additional skills found.")
                    
                    st.divider()
                    
                    # Recommendation
                    st.subheader("💡 Recommendation")
                    recommendation = result.get('recommendation', 'No recommendation available')
                    overall_score = result.get('overall_score', 0)
                    
                    if overall_score >= 75:
                        st.success(f"**{recommendation}**")
                    elif overall_score >= 50:
                        st.warning(f"**{recommendation}**")
                    else:
                        st.error(f"**{recommendation}**")
                    
                else:
                    st.error("Failed to extract text from the resume.")
                    
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e)

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666;">
        AI Resume Screener - Automated Resume Screening System
    </div>
""", unsafe_allow_html=True)
