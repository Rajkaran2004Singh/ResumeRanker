import pdfplumber
import os
import spacy
nlp = spacy.load("en_core_web_sm")
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
import re
from collections import defaultdict
import streamlit as st
import subprocess
import sys

def extract_pdf_text(file):
    full_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def classify_sections(text):
    section_headers = {
        "Education": ["education", "academic background", "qualifications"],
        "Experience": ["experience", "work history", "employment", "professional experience"],
        "Skills": ["skills", "technical skills", "technologies", "tools"],
        "Projects": ["projects", "personal projects", "portfolio"],
        "Certifications": ["certifications", "certification", "courses", "training"],
        "Contact": ["contact", "personal information", "email", "phone"],
    }
    header_keywords = {section: [kw.lower() for kw in keywords] for section, keywords in section_headers.items()}
    sections = defaultdict(list)
    current_section = "Other"
    for line in text.splitlines():
        line_clean = line.strip().lower().rstrip(":")
        line_clean = re.sub(r"[^a-z\s]", "", line_clean)
        found_section = False
        for section, keywords in header_keywords.items():
            if any(re.fullmatch(rf"{kw}", line_clean) for kw in keywords):
                current_section = section
                found_section = True
                break
        if not found_section and line.strip():
            sections[current_section].append(line.strip())
    return dict(sections)

# skills would be nouns or proper nouns

def extract_skills_from_resume(text):
    # POS tags and noun chunks
    doc = nlp(text)
    skills = set()
    for token in doc:
        if token.pos_ in {"PROPN", "NOUN"} and not token.is_stop:
            skills.add(token.text.lower())
    for chunk in doc.noun_chunks:
        skills.add(chunk.text.lower())
    # Pattern-based skill matching
    pattern = r"(?:proficient in|experienced with|technologies|tools|skills|frameworks)(.*)"
    for line in text.lower().splitlines():
        match = re.search(pattern, line)
        if match:
            for skill in re.split(r",|\|", match.group(1)):
                skills.add(skill.strip())
    # Capitalized terms
    skills.update(re.findall(r"\b([A-Z][a-zA-Z0-9\+\#]*)\b", text))
    return set(s.lower() for s in skills if len(s) > 1)

def extract_skills_from_jd(jd_text):
    doc = nlp(jd_text)
    skills = set()
    for token in doc:
        if token.text[0].isupper() and not token.is_sent_start:
            skills.add(token.text.lower())
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            skills.add(token.text.lower())
    for chunk in doc.noun_chunks:
        skills.add(chunk.text.lower())
    return set(s.lower() for s in skills if len(s) > 1)

def compute_similarity(resume_text, jd_text):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    similarity = util.cos_sim(resume_emb, jd_emb)
    return similarity[0][0].item()

def compare_skills(resume_skills, jd_skills):
    matched = resume_skills.intersection(jd_skills)
    missing = jd_skills.difference(resume_skills)
    return matched, missing

def process_resume(file, jd_text):
    full_text = extract_pdf_text(file)
    sections = classify_sections(full_text)
    resume_text = "\n".join(sections.get("Skills", [])) or full_text
    resume_skills = extract_skills_from_resume(full_text)
    jd_skills = extract_skills_from_jd(jd_text)
    score = compute_similarity(resume_text, jd_text)
    matched, missing = compare_skills(resume_skills, jd_skills)
    return {
        "file": file.name,
        "score": round(score, 2),
        "matched_skills": matched,
        "missing_skills": missing,
    }


def rank_resumes(folder_path, jd_text):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            result = process_resume(path, jd_text)
            results.append(result)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# Streamlit UI
st.title("Resume Matcher")
st.write("Upload multiple resumes and provide a job description. Get similarity scores in seconds!")

# Inputs
jd_text = st.text_area("Job Description")

uploaded_files = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)

if st.button("Match Resumes"):
    if not jd_text.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        st.success("Processing...")
        results = []

        for file in uploaded_files:
            result = process_resume(file, jd_text)
            results.append(result)


        # Display Results
        st.subheader("Match Results")
        for res in sorted(results, key=lambda x: -x["score"]):
            st.write(f"**{res['file']}**")
            st.write(f"Similarity Score: **{res['score']}**")
            st.write(f"Matched Skills: {', '.join(res['matched_skills'])}")
            st.write(f"Missing Skills: {', '.join(res['missing_skills'])}")
            st.markdown("---")
