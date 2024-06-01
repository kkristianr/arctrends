
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim

import streamlit as st
import streamlit.components.v1 as components

from os import path
import re

import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

st.set_page_config(page_title="Design shifts in healthcare", page_icon="favicon.ico", layout="wide", initial_sidebar_state="collapsed", menu_items=None)


papers = pd.read_csv('data/papers.csv')
data = papers.iloc[:, 4]
stopwords = ["et", "figure", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


@st.cache_data
def preprocess_papers (content):
    #content_str = ' '.join(content)
    lines = content.split('\n')
    
    ## TODO: check if both are needed. First one doesn't work properly? 
    new_lines = ''
    for line in lines:
        if line.endswith('-'):
            new_lines += line[:-1] + lines[lines.index(line)+1] + ' '
            lines.remove(lines[lines.index(line)+1])
        else:
            new_lines += line + ' '
    
    def remove_hyphenation(text):
        hyphenated_word_pattern = r'(\b\w+)-\s+(\w+\b)'
        
        def join_hyphenated(match):
            return match.group(1) + match.group(2)
        
        result = re.sub(hyphenated_word_pattern, join_hyphenated, text)
        
        return result

    new_lines = remove_hyphenation(new_lines)
    # tokenize words
    new_lines = word_tokenize(new_lines)

    # remove stopwords
    new_lines = [word for word in new_lines if word.lower() not in stopwords]

    #join two tokens if one ends with -
    new_lines = [word if not word.endswith('-') else word[:-1] + new_lines[new_lines.index(word)+1] for word in new_lines]

    new_lines = [re.sub('[^A-Za-z0-9-]+', '', word) for word in new_lines]
    #remove empty strings
    new_lines = [word for word in new_lines if word]

    return new_lines


articles_by_decade = papers.groupby('Decade')['Content'].apply(lambda x: ' '.join(x)).reset_index()
articles_by_decade['Content'] = articles_by_decade['Content'].astype("string")

articles_by_decade['Content'] = articles_by_decade['Content'].apply(preprocess_papers)




### USERFACE
 
st.title("Topic modeling")
# four columns layout
col1, col2, col3, col4 = st.columns((1,1,1,1))
with col1:
    k_topics = st.slider("Number of topics", 2, 10, value=5)


with st.spinner('Extracting the topics...'):
    dictionary = corpora.Dictionary(articles_by_decade['Content'])
    st.write("Dictionary size: " + str(len(dictionary)))
    corpus = [dictionary.doc2bow(text) for text in articles_by_decade['Content']]
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=k_topics, id2word=dictionary, passes=30, random_state=42, alpha='auto', per_word_topics=True)
    lda_model.save('models/lda.model')

with st.spinner('Visualizing the topics...'):
    lda_display = gensim.models.ldamodel.LdaModel.load('models/lda.model')
    lda_data =  pyLDAvis.gensim.prepare(lda_display, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(lda_data)
    components.html(html_string, width=1300, height=1200, scrolling=True)
