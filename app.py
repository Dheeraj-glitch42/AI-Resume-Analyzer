import streamlit as st # creat web page
import matplotlib.pyplot as plt # graph or charts
from sklearn .feature_extraction.text import TfidfTransformer # use for ML ,collection of text to numeric value
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity # similarty b/w vectors
import PyPDF2  # works on pdf like read , merge ,rotate,extract data etc
import re  # regular expression,  searching tool like patterns ,string ,words etc
from collections import Counter #count the words in list
import nltk # natural language toolkit, 
from nltk.corpus import stopwords # is ,are ,the etc are the words connects the sentence
from nltk.tokenize import word_tokenize # split into tokens 
from nltk import pos_tag #part of speech,assign levels  to sentences like noun pronoun adjective ..etc

# Download NLTK resourses


# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")


    
     #PAGE SETUP

st.set_page_config(page_title="Resume Job Match Scorer",page_icon='📝',layout="wide")  

st.markdown(""" 
Upload your resume (PDF) and paste a job description to see how well they match !   
 This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job 
requirements.            
 """)

with st.sidebar:
    st.header("About")
    st.info(""" 
This tool help you:
- Measures how your resume matches a job description
- Identify important job keywords
- Improve your resume based on missing terms
     """)
    

    st.header("How it works")
    st.write("""
1. Upload your resume (PDF)
2. Paste the job description
3. Click **Analyze Match**
4. Get similarity score
""")
    


          #  Helper functions

def extract_text_from_pdf(uploaded_file):
     try:
        pdf_reader=PyPDF2.PdfReader(uploaded_file)
        text=""
        for page in pdf_reader.pages:
             text=text+page.extract_text()
        return text
     except Exception as e:
          st.error(f"Error readin PDF:{e}")
          return ""             
     
#remove unwanted data and convert higher case to lower cases
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text) # sub(pattern,replacement,text), ^ means not 
    text=re.sub(r'\s+',' ',text).strip() # strip() means trim
    return text         
    

def remove_stopwords(text): # remove_stopwords is function name
    stop_words=set(stopwords.words('english'))     #stopwords is library that stores is am are etc
    # words=word_tokenize(text)
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])


# def calculate_similarity(resume_text,job_description):
#     resume_processed=remove_stopwords(clean_text(resume_text))
#     job_processed=remove_stopwords(clean_text(job_description))
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix=vectorizer.fit_transform([resume_processed,job_processed])
#     score=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:2])[0][0]*100
#     return round(score,2),resume_processed,job_processed

def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),   # words + phrases
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    return round(score, 2), resume_processed, job_processed

    
# def extract_keywords(text,num_keywords=10):
#     words=word_tokenize(text)
#     words=[w for w in words if len(w)>2]
#     tagged_words=pos_tag(words)
#     nouns=[w for w , pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
#     word_freq=Counter(nouns)
#     return word_freq.most_common(num_keywords)



              # MAIN APP

def main():
    uploaded_file=st.file_uploader("Upload your resume(PDF) ",type=['pdf'])
    job_description=st.text_area("Paste the job descroption",height=200)

    if st.button("Analyze Match"):
        if not uploaded_file:
            st.warning("Please upload your resume.")
            return
        if not job_description:
            st.warning("Please paste/write job description") 
            return  

        with st.spinner("Analyzing your resume...."):
            resume_text=extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Could not extract text from PDF. Please try another PDF")
                return
            # calculate similarity
            similarity_score,resume_processed,job_processed=calculate_similarity(resume_text,job_description)

            #Result
            st.subheader("Results")
            st.metric("Match Score",f"{similarity_score:.2f}%")

            # gauge chart

            fig,ax=plt.subplots(figsize=(6,0.5))
            colors=['#ff4b4b','#ffa726','#0f9d58']
            color_index=min(int(similarity_score//33),2)
            ax.barh([0],[similarity_score],color=colors[color_index])
            ax.set_xlim(0,100)
            ax.set_xlabel("Match percentage")
            ax.set_yticks([])
            ax.set_title("Resume Job Match")
            st.pyplot(fig)

            if similarity_score<40:
                st.warning("Low Match , consider tailoring your resume more closely.")
            elif similarity_score<70:
                st.info("Good Match , Your resume aligh fairly well.")
            else:
                 st.success("Excellent Match ! Your resume strongly aligns.")       
     
   

if __name__ == "__main__":
    main()
        

    


