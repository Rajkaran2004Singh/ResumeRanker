# ResumeRanker
# NLP-Based Resume Matcher

This project uses Natural Language Processing (NLP) and semantic similarity techniques to match resumes for a given job description. It helps recruiters to shortlist candidates more efficiently by scoring and ranking resumes based on skill overlap and contextual relevance.

# Features

1. Extracts text from multiple PDF resumes
2. Parses resumes to classify key sections like **Skills**, **Experience**, etc.
3. Uses NLP to extract **skills** from both the job description and resumes
4. Calculates semantic similarity scores using **BERT-based embeddings**
5. Identifies **matched** and **missing skills**
6. Ranks candidates based on relevance
7. Interactive web interface to upload resumes and input JD

# Technologies Used

- **Language:** Python  
- **NLP:** spaCy, Sentence-Transformers (MiniLM)  
- **ML Backend:** PyTorch (for BERT embeddings)  
- **PDF Parsing:** pdfplumber  
- **Web App:** Streamlit (only for UI rendering)  


# NLP Pipeline

1. Text Extraction : Resumes are parsed using **pdfplumber**.
2. Section Classification: Segments resumes into logical sections using keyword-based heuristics.
3. Skill Extraction :
   a. Extracts nouns, proper nouns, and noun chunks using `spaCy`.
   b. Matches pattern-based phrases like "Proficient in", "Experienced with", etc.
4. Semantic Similarity :
   a. Encodes text using BERT (all-MiniLM-L6-v2) from sentence-transformers.
   b. Computes cosine similarity to determine the JD-resume match score.
5. Skill Comparison : Identifies overlap and gaps between JD skills and candidate skills.

# App Hosted 
https://resumeranker-streamlt0.streamlit.app

