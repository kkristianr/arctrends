
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

import streamlit as st
from os import path
from PIL import Image
from collections import Counter
import re
import json
from io import StringIO 
from wordcloud import WordCloud, ImageColorGenerator
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings


import multiprocessing
import sklearn.manifold

st.set_page_config(page_title="Design shifts in healthcare", page_icon="favicon.ico", layout="wide", initial_sidebar_state="collapsed", menu_items=None)

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

def visualize_word2vec(model):
    vocab = list(model.wv.key_to_index)
    X = model.wv[vocab]
    
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    
    df = pd.DataFrame(X_2d, index=vocab, columns=['x', 'y'])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])
    
    for word, pos in df.iterrows():
        ax.annotate(word, pos)
    
    st.pyplot(fig)

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

stemmer = PorterStemmer()

def get_stem(query):
    return stemmer.stem(query.lower())

def process_query(query):
    tokens = word_tokenize(query)
    if len(tokens) == 1:
        return stemmer.stem(query.lower()), True
    else:
        return query.lower(), False


def search_query(data, query, is_single_word, words_before=50, words_after=50):
    results = []
    for index, row in data.iterrows():
        content = row['Content']
        tokens = word_tokenize(content)
        paper_results = []

        if is_single_word:
            for i, token in enumerate(tokens):
                if stemmer.stem(token.lower()) == query:
                    start = max(0, i - words_before)
                    end = min(len(tokens), i + words_after + 1)
                    before = ' '.join(tokens[start:i])
                    after = ' '.join(tokens[i+1:end])
                    match = tokens[i]
                    paper_results.append((before, match, after))
        else:
            content_lower = content.lower()
            query_len = len(query)
            start = 0
            while start < len(content_lower):
                start = content_lower.find(query, start)
                if start == -1:
                    break
                end = start + query_len
                before_start = max(0, start - (words_before*8))
                after_end = min(len(content), end + (words_after*8))
                before = content[before_start:start]
                after = content[end:after_end]
                match = content[start:end]
                paper_results.append((before, match, after))
                start = end  # Continue searching after the current match
        
        if paper_results:
            results.append((row['Title'], row['Year'], paper_results))
    return results


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

results_tab, eval_tab, clouds_tab, terms_tab = st.tabs(["Visualize shifts","Eval", "Word clouds","Terms of interest"])


with results_tab:
        #try:
         #   model = load_model('models/word2vec.model')
        #except FileNotFoundError:
         #   st.error("Model not found. Please train the model first. Open the sidebar on the left-side <- ")
          #  st.stop()     """
    
    #visualize_word2vec(model)
    
    articles_by_year = papers.groupby('Year')['Content'].apply(lambda x: ' '.join(x)).reset_index()
    articles_by_year["Content"] = preprocess_papers(articles_by_year["Content"], stop_words = stop_words_bool, bigrams = bigrams_bool, trigrams = trigrams_bool)
    articles_by_year["Content"] = articles_by_year["Content"].apply(lambda x: ' '.join(x))
    
    terms = load_terms('terms.json')
    distances_df = pd.DataFrame(columns=['main term', 'year','related term', 'distance'])
    articles_by_year['Year'] = articles_by_year['Year'].astype(int)
    distances_df.drop(distances_df.index, inplace=True)
    #distances_df= pd.read_csv('distances.csv')
    
    for year, dataset in zip(articles_by_year['Year'], articles_by_year['Content']):
        model = train_word2vec(data, stop_words_bool, bigrams_bool, trigrams_bool, vector_dim, window_length, min_count_word)
        model.save(f'models/word2vec_{year}.model')
        for main_term, related_terms in terms.items():
            for related_term in related_terms:
                try:
                    cosine_distance = model.wv.similarity(main_term, related_term)
                except KeyError:
                    cosine_distance = None  

                distances_df = pd.concat([distances_df, pd.DataFrame({'main term': [main_term], 'year': [year], 'related term': [related_term], 'distance': [cosine_distance]})])
    distances_df['year'] = distances_df['year'].astype(int)
    distances_df.to_csv('distances.csv', index=False)

    main_terms = distances_df['main term'].unique()

    import altair as alt

    for main_term in main_terms:
        main_data = distances_df[distances_df['main term'] == main_term]
        chart = alt.Chart(main_data).mark_line().encode(
            x='year:O',
            y='distance',
            color='related term',
            strokeDash='related term',
        )
        st.altair_chart(chart, theme=None, use_container_width=True)


with eval_tab:
    st.title('Contextualized search')
    query = st.text_input('Enter a word to search:')

    if query:
        processed_query, is_single_word = process_query(query)
        results = search_query(papers, processed_query, is_single_word)

        if results:
            st.write(f'Found {sum(len(paper_results) for _, _, paper_results in results)} results for "{query}"')
            
            for title, year, paper_results in results:
                with st.expander(f'{title} ({year}) - {len(paper_results)} matches'):
                    for before, match, after in paper_results:
                        st.write(f'...{before}  :red[**{match}**] {after} ...')
        else:
            st.write(f'No results found for "{query}"')

with clouds_tab: 
    articles_by_year = papers.groupby('Year')['Content'].apply(lambda x: ' '.join(x)).reset_index()
    articles_by_year["Content"] = preprocess_papers(articles_by_year["Content"], stop_words = True, bigrams = False, trigrams = False)
    articles_by_year["Content"] = articles_by_year["Content"].apply(lambda x: ' '.join(x))
    
    st.write("## Word clouds for every year (later decade)")
    st.write("The word clouds show the most frequent words in the papers for each year.")


    for year, dataset in zip(articles_by_year['Year'], articles_by_year['Content']):
        tokens_count = len(dataset.split())
        st.write(f"Year: {year}, Number of tokens: {tokens_count}")

        wordcloud = WordCloud(width=1600, 
                            height=800,    
                            background_color='white',
                            collocations = False,     
                            max_words = 150, 
                            max_font_size=150, 
                            stopwords = stopwords, 
                            random_state=42).generate(dataset)
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(year)
        st.pyplot(plt)

    combined_content = ' '.join(articles_by_year['Content'])
    word_counts_total = Counter(combined_content.split())
    relevant_terms = np.unique(distances_df[['main term','related term']].values)
    
    def compute_word_count(year_content):
        word_counts_year = Counter(year_content.split())
        return [word_counts_year[word] for word in relevant_terms]

    word_counts_by_year = articles_by_year['Content'].apply(compute_word_count).tolist()
    word_counts_df = pd.DataFrame(word_counts_by_year, columns=relevant_terms)
    plt.figure(figsize=(12, 20))
    num_years = len(articles_by_year)
    bar_height = 0.25

    for i, year in enumerate(articles_by_year['Year']):
        positions = range(len(relevant_terms))
        plt.barh([pos + i * bar_height for pos in positions], word_counts_df.iloc[i], bar_height, label=year)

    plt.yticks([pos + (len(articles_by_year) - 1) * bar_height / 2 for pos in positions], relevant_terms)
    plt.xlabel('Count')
    plt.title('Count of relevant terms')
    plt.legend()
    plt.tight_layout()

    st.pyplot(plt)

    st.write("## Terms generator")
    st.write("Enter a term to generate a list of its neighbours.")
    term = st.text_input("Enter term:")
    term = format_term(term)
    if term is not None:
        try:
            model = load_model('models/word2vec.model')
        except FileNotFoundError:
            st.error("Model not found. Please train the model first.")
            st.stop()
        if st.button("Generate"):
            neighbours = list_neighbours(model, term, topn=100)
            neighbours = [word for word, _ in neighbours]
            # List main term and its neighbours with text
            st.write(f"Main term: {term}")
            st.write(f"Neighbours: {', '.join(neighbours)}")



            neighbours = " ".join(neighbours)
            wordcloud = WordCloud(width=1600, 
                                height=800,                   
                                max_words = 150, 
                                max_font_size=150, 
                                stopwords = stopwords, 
                                random_state=42).generate(neighbours)
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Neighbours of {term}")
            st.pyplot(plt)

with terms_tab:

    file_path = 'terms.json'
    try:
        terms = load_terms(file_path)
        st.write("### Terms of interest")
        table_terms= {'Main term': list(terms.keys()),
                        'Related terms': [", ".join(terms[main_term]) for main_term in terms.keys()]}
        st.table(table_terms)

        
    except FileNotFoundError:
        st.error("File not found.")

    action = st.selectbox("Select action", ["Add main term", "Edit main term", "Delete main term", "Add related term", "Delete related term"])

    if action == "Add main term":
        new_main_term = st.text_input("Enter new main term:")
        new_main_term = format_term(new_main_term)
        if st.button("Add"):
            terms[new_main_term] = []
            save_terms(terms, file_path)
            st.success("Main term added successfully.")

    elif action == "Edit main term":
        main_term_to_edit = st.selectbox("Select main term to edit:", list(terms.keys()))
        edited_main_term = st.text_input("Enter edited main term:", value=main_term_to_edit)
        edited_main_term = format_term(edited_main_term)
        if st.button("Edit"):
            terms[edited_main_term] = terms.pop(main_term_to_edit)
            save_terms(terms, file_path)
            st.success("Main term edited successfully.")

    elif action == "Delete main term":
        main_term_to_delete = st.selectbox("Select main term to delete:", list(terms.keys()))
        if st.button("Delete"):
            del terms[main_term_to_delete]
            save_terms(terms, file_path)
            st.success("Main term deleted successfully.")

    elif action == "Add related term":
        main_term = st.selectbox("Select main term:", list(terms.keys()))
        new_related_term = st.text_input("Enter new related term:")
        new_related_term = format_term(new_related_term)
        if st.button("Add"):
            terms[main_term].append(new_related_term)
            save_terms(terms, file_path)
            st.success("Related term added successfully.")

    elif action == "Delete related term":
        main_term = st.selectbox("Select main term:", list(terms.keys()))
        related_term_to_delete = st.selectbox("Select related term to delete:", terms[main_term])
        if st.button("Delete"):
            terms[main_term].remove(related_term_to_delete)
            save_terms(terms, file_path)
            st.success("Related term deleted successfully.")

