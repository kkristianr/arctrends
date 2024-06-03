import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json
import os
import re

from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser
import nltk
from nltk.tokenize import sent_tokenize
import multiprocessing
import altair as alt

nltk.download('punkt')

st.set_page_config(page_title="Design shifts in healthcare", page_icon="favicon.ico", layout="centered", initial_sidebar_state="expanded", menu_items=None)

st.write("## Trends in healthcare architectural design")


def preprocess_papers (content, stop_words = True, bigrams = True, trigrams = False):
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

    sentences = sent_tokenize(new_lines)

    # tokenize sentences
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    #join two tokens if one ends with -
    sentences = [[word if not word.endswith('-') else word[:-1] + sentences[sentences.index(sentence)+1][0] for word in sentence] for sentence in sentences]

    #stopwords
    if stop_words:
        sentences = [[word for word in sentence if word.lower() not in stopwords] for sentence in sentences]

    sentences = [[re.sub('[^A-Za-z0-9-]+', '', word) for word in sentence] for sentence in sentences]
    #remove empty strings
    sentences = [[word for word in sentence if word] for sentence in sentences]

    if bigrams:
        bigram = Phrases(sentences, min_count=5, threshold=1)
        sentences = [bigram[sentence] for sentence in sentences]
    
    if trigrams:
        trigram = Phrases(bigram[sentences], min_count=5, threshold=1)
        sentences = [trigram[bigram[sentence]] for sentence in sentences]
    

    return sentences
    
def train_word2vec(data, vector_dim, window_length, min_count): 
    model = Word2Vec(sentences = data, vector_size=vector_dim, sg=1, negative=10, window=window_length, min_count=min_count, workers=multiprocessing.cpu_count())
    print("Word2Vec vocabulary length:", len(model.wv.key_to_index))
    model.train(data, total_examples=len(data), epochs=25)
    return model


def save_model(model, path):
    model.save(path)
    print("Model saved")

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


papers = pd.read_csv('data/papers.csv')
#data = papers.iloc[:, 4]
stopwords = ["et", "figure", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
articles_by_decade = papers.groupby('Decade')['Content'].apply(lambda x: ' '.join(x)).reset_index()
articles_by_decade['Content'] = articles_by_decade['Content'].astype("string")


##data = ' '.join(articles_by_decade['Content'])
#data = preprocess_papers(data, stop_words = True, bigrams = True, trigrams = False)
#model = train_word2vec(data, 300, 30)
#model.save('models/word2vec.model')
#print("Model saved")


terms = load_terms('terms.json')
distances_df = pd.DataFrame(columns=['main term', 'decade','related term', 'distance'])
#distances_df.drop(distances_df.index, inplace=True)


st.sidebar.header("Text preprocessing settings")
stop_words_bool = st.sidebar.checkbox("Remove stop words", False)
bigrams_bool = st.sidebar.checkbox("Include bigrams", True)
trigrams_bool = st.sidebar.checkbox("Include trigrams", True)
st.sidebar.header("Model settings")
min_count = st.sidebar.slider("Minimum count",1,10, value=2)
window_length = st.sidebar.slider("Window length",2,30, value=20)
vector_dim = st.sidebar.slider("Vectors dimension",100,300, value=100, step=50)
if st.sidebar.button("Re-train the models"):
    with st.spinner("Training the models..."):
        for decade, dataset in zip(articles_by_decade['Decade'], articles_by_decade['Content']):
            dataset = preprocess_papers(dataset, stop_words = stop_words_bool, bigrams = bigrams_bool, trigrams = trigrams_bool)
            model = train_word2vec(dataset, vector_dim, window_length, min_count)
            print(model)

            model.save(f'models/word2vec_{decade}.model')

for decade, dataset in zip(articles_by_decade['Decade'], articles_by_decade['Content']):
    dataset = preprocess_papers(dataset, stop_words = stop_words_bool, bigrams = bigrams_bool, trigrams = trigrams_bool)
    if os.path.exists(f'models/word2vec_{decade}.model'):
        with st.spinner('Loading the models...'):
            model = load_model(f'models/word2vec_{decade}.model')
    else:
        with st.spinner('Training the models...'):
            model = train_word2vec(dataset, vector_dim, window_length, min_count)
            model.save(f'models/word2vec_{decade}.model')

    for main_term, related_terms in terms.items():
        for related_term in related_terms:
            try:
                cosine_distance = model.wv.similarity(main_term, related_term)
                print(decade, main_term, related_term, cosine_distance)
            except KeyError:
                cosine_distance = None  
            distances_df = pd.concat([distances_df, pd.DataFrame({'main term': [main_term], 'decade': [decade], 'related term': [related_term], 'distance': [cosine_distance]})])
distances_df.to_csv('distances.csv', index=False)

with st.spinner('Creating the graphs...'):
    distances_df = load_distances('distances.csv')
    main_terms = distances_df['main term'].unique()

    for main_term in main_terms:
        main_data = distances_df[distances_df['main term'] == main_term]
        st.text(f" Shift in similarity with respect to: {main_term}")
        chart = alt.Chart(main_data).mark_line(
            #point=alt.OverlayMarkDef(filled=False, fill="white")
        ).encode(
            x=alt.X('decade:N', title='', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('distance', title='Cosine similarity'),
            color='related term',
            strokeDash='related term',
            tooltip=['related term', 'distance']  
        ).interactive()

        point = alt.Chart(main_data).mark_point(size=50, filled=True).encode(
            x='decade:N',
            y='distance',
            color='related term',
            tooltip=['related term', 'distance']
        ).interactive()

        st.altair_chart(chart + point, theme='streamlit', use_container_width=True)


