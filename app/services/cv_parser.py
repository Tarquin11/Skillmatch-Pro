import pdfplumber 
import io

#example 

KNOWN_SKILLS = ["Python", "Java", "C++", "SQL", "Machine Learning", "Data Analysis", "Project Management"]

def extract_text(file_bites, filename):
        if filename.endswith('.pdf'):
            with pdfplumber.open(io.BytesIO(file_bites)) as pdf:
              text = ""
              for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        
        return ""
skills_db = [s.lower() for s in KNOWN_SKILLS]

def detect_skills(text):
    text = text.lower()
    return [skill for skill in skills_db if skill in text]
