import streamlit as st
st.set_page_config(page_title="Resume Detection",layout="wide",initial_sidebar_state='collapsed')

import pickle
import nltk
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download('punkt')


#loading model
clf=pickle.load(open('clf.pkl','rb'))
tfid=pickle.load(open('tfidf.pkl','rb'))


stop_words = set(stopwords.words("english"))

def clean_text(text):
    ps = PorterStemmer()
    
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = text.lower()
    text = text.split()
    
    text = [ps.stem(word) for word in text if word not in stop_words]
    
    return ' '.join(text)

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}





st.title('Resume Screening App')
uploaded_files = st.file_uploader( 
    "Upload Resume",type=['txt','pdf'] ,accept_multiple_files=False
)
if uploaded_files is not None:
    try:
        resume_bytes=uploaded_files.read()
        resume_text=resume_bytes.decode('utf-8')
    except UnicodeDecodeError:
        resume_text=resume_bytes.decode('latin-1')

    clean_resume=clean_text(resume_text)
    clean_resume=tfid.transform([clean_resume])
    prediction_id=clf.predict(clean_resume)[0]
    category_name=category_mapping.get(prediction_id,"unknown")

    st.write("🙌💼🎯 Predicted Category:",category_name)



