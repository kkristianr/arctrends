import pandas as pd
import streamlit as st

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Design shifts in healthcare", page_icon="favicon.ico", layout="wide", initial_sidebar_state="collapsed", menu_items=None)

papers = pd.read_csv('data/sust.csv')
data = papers.iloc[:, 4]

stopwords = ["et", "figure", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

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



st.write('## Contextualized search')
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