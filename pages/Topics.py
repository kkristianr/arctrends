
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pyLDAvis
import pyLDAvis.gensim

import pymupdf
from multi_column import column_boxes

import streamlit as st
import streamlit.components.v1 as components

from os import path
from PIL import Image
from collections import Counter
import re
import json
from io import StringIO 
from wordcloud import WordCloud, ImageColorGenerator
import gensim
from gensim import corpora
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser
import nltk
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Design shifts in healthcare", page_icon="favicon.ico", layout="wide", initial_sidebar_state="collapsed", menu_items=None)


papers = pd.read_csv('data/sust.csv')
data = papers.iloc[:, 4]
stopwords = ["et", "figure", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

articles_by_year = papers.groupby('Year')['Content'].apply(lambda x: ' '.join(x)).reset_index()

@st.cache_data
def preprocess_papers (papers, stop_words = True, stemming = False, bigrams = True, trigrams = False):
    # Tokenize the papers
    tokenized_papers = [word_tokenize(paper.lower()) for paper in papers]
    
    # Remove digits and special characters
    tokenized_papers = [[word for word in paper if word.isalpha()] for paper in tokenized_papers]
    
    # Remove words starting with http and www
    tokenized_papers = [[word for word in paper if not word.startswith('http') and not word.startswith('www')] for paper in tokenized_papers]

    if stop_words:
        tokenized_papers = [[word for word in paper if word not in stopwords] for paper in tokenized_papers]
    
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        tokenized_papers = [[stemmer.stem(word) for word in paper] for paper in tokenized_papers]
    
    # Bigrams
    if bigrams:
        bigram_phrases = Phrases(tokenized_papers)
        bigram = Phraser(bigram_phrases)
        tokenized_papers = [bigram[paper] for paper in tokenized_papers]
    
    # Trigrams
    if trigrams:
        trigram_phrases = Phrases(bigram[tokenized_papers])
        trigram = Phraser(trigram_phrases)
        tokenized_papers = [trigram[bigram[paper]] for paper in tokenized_papers]
    
    return tokenized_papers


### USERFACE
 

k_topics = st.slider("Number of topics", 2, 10, value=3)
articles_by_year["Content"] = preprocess_papers(articles_by_year["Content"], stop_words = True, bigrams = True, trigrams = True)

with st.spinner('Extracting the topics...'):
    dictionary = corpora.Dictionary(articles_by_year['Content'])
    corpus = [dictionary.doc2bow(text) for text in articles_by_year['Content']]
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=k_topics, id2word=dictionary, passes=30, random_state=42, alpha='auto', per_word_topics=True)
    lda_model.save('models/lda.model')

with st.spinner('Visualizing the topics...'):
    lda_display = gensim.models.ldamodel.LdaModel.load('models/lda.model')
    lda_data =  pyLDAvis.gensim.prepare(lda_display, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(lda_data)
    components.html(html_string, width=1300, height=1200, scrolling=True)
