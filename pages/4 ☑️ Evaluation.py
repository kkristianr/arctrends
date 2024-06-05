import pandas as pd
import streamlit as st
import altair as alt
import re

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models.phrases import Phrases

st.set_page_config(page_title="Design shifts in healthcare", page_icon="favicon.ico", layout="wide", initial_sidebar_state="expanded", menu_items=None)

stopwords = ["et", "figure", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
stemmer = PorterStemmer()

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

    #join two tokens if one ends with -
    new_lines = [word if not word.endswith('-') else word[:-1] + new_lines[new_lines.index(word)+1] for word in new_lines]

    new_lines = [re.sub('[^A-Za-z0-9-]+', '', word) for word in new_lines]
    #remove empty strings
    new_lines = [word for word in new_lines if word]

    #stem words
    #new_lines = [stemmer.stem(word) for word in new_lines]

    # bigrams
    bigram = Phrases(new_lines, min_count=5, threshold=0.5)
    new_lines = bigram[new_lines] 
    

    trigram = Phrases(bigram[new_lines], min_count=3, threshold=0.5)
    new_lines = trigram[bigram[new_lines]]     

    return new_lines


papers = pd.read_csv('data/papers.csv')
articles_by_decade = papers.groupby('Decade')['Content'].apply(lambda x: ' '.join(x)).reset_index()
articles_by_decade['Content'] = articles_by_decade['Content'].astype("string")
for decade, dataset in zip(articles_by_decade['Decade'], articles_by_decade['Content']):
            dataset = preprocess_papers(dataset)

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


def compute_distances(main_term, related_term):
    main_term = format_term(main_term)
    related_term = format_term(related_term)
    main_term_indexes = []
    related_term_indexes = []
    main_term_list = []
    related_term_list = []
    distance_list = []
    decade_list = []
    topic_positions = []
    term_positions = []
    for decade, dataset in zip(articles_by_decade['Decade'], articles_by_decade['Content']):
        dataset = preprocess_papers(dataset)
        for i, word in enumerate(dataset):
            if word == main_term:
                main_term_indexes.append((decade, i))
            if word == related_term:
                related_term_indexes.append((decade, i))
        for main_decade, main_index in main_term_indexes:
            for related_decade, related_index in related_term_indexes:
                if main_decade == related_decade:
                    distance = abs(main_index - related_index)
                    decade_list.append(main_decade)
                    main_term_list.append(main_term)
                    topic_positions.append(main_index)
                    related_term_list.append(related_term)
                    term_positions.append(related_index)
                    distance_list.append(distance)
        results = pd.DataFrame({'Decade': decade_list, 'Main term': main_term_list, 'Main index': topic_positions, 'Related term': related_term_list, 'Term index': term_positions ,'Distance': distance_list})
        results['Distance'] = results['Distance'].astype(int)
    return results

def get_shortest_distance(query, related_term):
    distances = compute_distances(query, related_term)
    # get the shortest distance for every main index
    shortest_distances = distances.groupby(['Decade','Main index'])['Distance'].min().reset_index()
    return shortest_distances



def get_stem(query):
    return stemmer.stem(query.lower())

def process_query(query):
    tokens = word_tokenize(query)
    if len(tokens) == 1:
        return stemmer.stem(query.lower()), True
    else:
        return query.lower(), False

def search_query(data, query, is_single_word, context_length=50):
    results = []
    for index, row in data.iterrows():
        content = row['Content']
        tokens = word_tokenize(content)
        paper_results = []

        if is_single_word:
            for i, token in enumerate(tokens):
                if stemmer.stem(token.lower()) == query:
                    start = max(0, i - context_length)
                    end = min(len(tokens), i + context_length + 1)
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
                before_start = max(0, start - (context_length*6))
                after_end = min(len(content), end + (context_length*6))
                before = content[before_start:start]
                after = content[end:after_end]
                match = content[start:end]
                paper_results.append((before, match, after))
                start = end  # Continue searching after the current match
        
        if paper_results:
            results.append((row['Title'], row['Year'], paper_results))
            #sort by year 
            results.sort(key=lambda x: x[1])
    return results

#sidebar 
st.sidebar.title('Search options')

query = st.sidebar.text_input('Enter a topic to search:')
context_length = st.sidebar.slider('Context length:', 0, 200, 50, help='Number of words before and after the match')

if query:
    processed_query, is_single_word = process_query(query)
    results = search_query(papers, processed_query, is_single_word, context_length)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write('## Matches')
        if results:
            st.write(f'Found {sum(len(paper_results) for _, _, paper_results in results)} results for "{query}"')

            related_term = st.text_input('Search for related term in results: (optional)')

            for title, year, paper_results in results:
                paper_count = 0
                with st.expander(f'{title} ({year}) - {len(paper_results)} matches'):
                    for before, match, after in paper_results:
                        stem_related_term = get_stem(related_term)
                        count = 0
                        for i in range(len(before.split())):
                            if get_stem(before.split()[i]) == stem_related_term:
                                count += 1
                                before = ' '.join([f':blue[**{word}**]' if get_stem(word) == stem_related_term else word for word in before.split()])
                                break
                        for i in range(len(after.split())):
                            if get_stem(after.split()[i]) == stem_related_term:
                                count += 1
                                after = ' '.join([f':blue[**{word}**]' if get_stem(word) == stem_related_term else word for word in after.split()])
                                break

                        st.write(f'...{before}  :red[**{match}**] {after} ...')
                        paper_count += count
                    if related_term:
                        st.write(f'Found {paper_count} matches for "{related_term}" in this paper.')
        
            with col2:
                if related_term:
                    st.write('## Distances')
                    st.write(f'Distance in tokens (=words) between {query} and {related_term}')
                    distances = compute_distances(query, related_term)
                    min_distance = get_shortest_distance(query, related_term)
                    print(min_distance)
                    if distances is not None:
                        chart = alt.Chart(min_distance).mark_bar(
                            opacity=0.3,
                            binSpacing=0
                        ).encode(
                            alt.X('Distance:Q', bin=alt.Bin(maxbins=50)),
                            alt.Y('count()', stack=None),
                            alt.Color('Decade:N')
                        )


                        st.altair_chart(chart, theme="streamlit", use_container_width=True)
                    

                        chart2 = alt.Chart(min_distance).transform_density(
                            'Distance',
                            as_=['Distance', 'Density'],
                            groupby=['Decade']
                        ).mark_area(
                        ).encode(
                            alt.X('Distance:Q', title='Distance'),
                            alt.Y('Density:Q', title='Density'),
                            alt.Row('Decade:N')
                        ).properties(height=100)


                        st.altair_chart(chart2, theme="streamlit", use_container_width=False
                        )       

        else:
            st.write(f'No results found for "{query}"')
else:
    st.write("👈 Use the sidebar to search for a topic")

