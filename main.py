# Import all the nessary library most notabley nltk and sklearn which is doing the main scoring calculation 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Downloading the extra nltk resorces for 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


app = FastAPI()


# structuring class which is allowning to take input in required structured 
class pair(BaseModel):
    text1: str
    text2: str


# preprocessing function a custom function which pre-process our inputdata by which we can calculate similarity more efficientely
def preProcess(t):
    t = t.lower() # converting entire input text into lower casing 

    tokened = word_tokenize(t) # tokenizing our text

    stop_words = set(stopwords.words('english')) # stop word agent
    lemmatizer = WordNetLemmatizer() # lemmatizing agent 

    tokened = [word for word in tokened if word not in stop_words] # removing all the stop words
    tokened = [lemmatizer.lemmatize(word) for word in tokened] # performing lemmatization


    ft = ' '.join(tokened) # after performing all of our process joing back the sentences in a final string

    return ft   #returning the final preprocessed text


# api end point "similarity" which will resend the similarity score
@app.post("/similarity")
async def similarity_score(text_pair: pair):

    ft1 = preProcess(text_pair.text1) # getting our pre-processed text 1
    ft2 = preProcess(text_pair.text2) # getting our pre-processed text 2

    vectorizer = TfidfVectorizer() # vectorizing agent
    tfidfMatrix = vectorizer.fit_transform([ft1,ft2]) # tfidf matrix

    similarity_matrix = cosine_similarity(tfidfMatrix) # calculating similarity score 
    similarity_score = similarity_matrix[0, 1]
    
    return {"similarity score": similarity_score} # sending back the final score.
