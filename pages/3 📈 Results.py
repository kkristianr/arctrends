import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from collections import Counter
import json

from wordcloud import WordCloud, ImageColorGenerator
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings

import multiprocessing
import sklearn.manifold
import altair as alt


st.set_page_config(page_title="Design shifts in healthcare", page_icon="favicon.ico", layout="centered", initial_sidebar_state="collapsed", menu_items=None)

st.write("## Trends in healthcare architectural design")



papers = pd.read_csv('data/sust.csv')
data = papers.iloc[:, 4]
stopwords = ["et", "figure", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


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


def train_word2vec(data, stop_words_bool, bigrams_bool, trigrams_bool, vector_dim, window_length, min_count_word): 
    tokenized_papers = preprocess_papers(data, stop_words = stop_words_bool, bigrams = bigrams_bool, trigrams = trigrams_bool)

    print("Data has been preprocessed")
    model = Word2Vec(sg=1, vector_size=vector_dim, seed=42, window=window_length, min_count=min_count_word, workers=multiprocessing.cpu_count())

    model.build_vocab(tokenized_papers)
    print("The vocabulary is built")

    print("Word2Vec vocabulary length:", len(model.wv.key_to_index))

    print("Entering training ")

    model.train(tokenized_papers, total_examples=model.corpus_count, epochs=30)
    print("Training finished")

    return model


def save_model(model, path):
    model.save(path)
    print("Model saved")

@st.cache_resource
def load_model(path):
    model = Word2Vec.load(path)
    print("Model loaded")
    return model

def load_terms(file_path = 'terms.json'):
    with open(file_path, 'r') as file:
        terms = json.load(file)
    return terms

def save_terms(terms, file_path):
    with open(file_path, 'w') as file:
        json.dump(terms, file, indent=4)

def format_term(term):
    words = term.strip().split()
    if len(words) > 3:
        st.error("Please enter a term with 1 to 3 words only.")
        return None
    elif len(words) > 1:
        n_words = len(words)
        term = "_".join(words)
        if n_words == 2:
            st.warning("The entered term is a bigram. It should frequently appear together in the dataset to be shown.")
        elif n_words == 3:
            st.warning("The entered term is a trigram. It should frequently appear together in the dataset to be shown.")
    return term


def list_neighbours(model, word, topn=50):
    neighbours = model.wv.most_similar(word, topn=topn)
    return neighbours

def load_distances(file_path = 'distances.csv'):
    distances = pd.read_csv(file_path)
    return distances

### USERFACE
    
st.sidebar.header("Text preprocessing settings")
stop_words_bool = st.sidebar.checkbox("Remove stop words", True)
bigrams_bool = st.sidebar.checkbox("Include bigrams", True)
trigrams_bool = st.sidebar.checkbox("Include trigrams", True)
st.sidebar.header("Model settings")
window_length = st.sidebar.slider("Window length",2,10, value=3)
vector_dim = st.sidebar.slider("Vectors dimension",100,300, value=100, step=50)
min_count_word=st.sidebar.slider("Minimum count of words threshold",3,20, value=3) 
if st.sidebar.button("Re-train the model"):
    st.write("Training the model...")
    model = train_word2vec(data, stop_words_bool, bigrams_bool, trigrams_bool, vector_dim, window_length, min_count_word)
    model.save('models/word2vec.model')






    #try:
        #   model = load_model('models/word2vec.model')
    #except FileNotFoundError:
        #   st.error("Model not found. Please train the model first. Open the sidebar on the left-side <- ")
        #  st.stop()     """

#visualize_word2vec(model)

articles_by_decade = papers.groupby('Decade')['Content'].apply(lambda x: ' '.join(x)).reset_index()
articles_by_decade["Content"] = preprocess_papers(articles_by_decade["Content"], stop_words = stop_words_bool, bigrams = bigrams_bool, trigrams = trigrams_bool)
articles_by_decade["Content"] = articles_by_decade["Content"].apply(lambda x: ' '.join(x))


terms = load_terms('terms.json')
distances_df = pd.DataFrame(columns=['main term', 'decade','related term', 'distance'])
distances_df.drop(distances_df.index, inplace=True)
#distances_df= pd.read_csv('distances.csv')

for decade, dataset in zip(articles_by_decade['Decade'], articles_by_decade['Content']):
    model = train_word2vec(data, stop_words_bool, bigrams_bool, trigrams_bool, vector_dim, window_length, min_count_word)
    model.save(f'models/word2vec_{decade}.model')
    for main_term, related_terms in terms.items():
        for related_term in related_terms:
            try:
                cosine_distance = model.wv.similarity(main_term, related_term)
            except KeyError:
                cosine_distance = None  

            distances_df = pd.concat([distances_df, pd.DataFrame({'main term': [main_term], 'decade': [decade], 'related term': [related_term], 'distance': [cosine_distance]})])
distances_df.to_csv('distances.csv', index=False)

main_terms = distances_df['main term'].unique()

for main_term in main_terms:
    main_data = distances_df[distances_df['main term'] == main_term]
    st.text(f" Shift in similarity to: {main_term}")
    chart = alt.Chart(main_data).mark_line().encode(
        x='decade:N',
        y='distance',
        color='related term',
        strokeDash='related term',
    )
    st.altair_chart(chart, theme=None, use_container_width=True)



