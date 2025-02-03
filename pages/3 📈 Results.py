import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json
import os
import re

from cade.cade import CADE
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser
import nltk
from nltk.tokenize import sent_tokenize
from nltk import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
import multiprocessing
import altair as alt

nltk.download('punkt')

st.set_page_config(page_title="Design shifts in healthcare", page_icon="favicon.ico", layout="centered", initial_sidebar_state="expanded", menu_items=None)

stopwords = ["et", "figure", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

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

    sentences = [[re.sub('[^A-Za-z-]+', '', word) for word in sentence] for sentence in sentences]
    #remove empty strings
    sentences = [[word for word in sentence if word] for sentence in sentences]

    if bigrams:
        bigram = Phrases(sentences, min_count=5, threshold=1)
        sentences = [bigram[sentence] for sentence in sentences]
    
    if trigrams:
        trigram = Phrases(bigram[sentences], min_count=5, threshold=1)
        sentences = [trigram[bigram[sentence]] for sentence in sentences]
    

    return sentences


def save_model(model, path):
    model.save(path)
    st.write("Model saved to", path)

def load_model(path):
    model = Word2Vec.load(path)
    st.write("Model loaded from", path)
    return model

def load_terms(file_path='terms.json'):
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

def load_distances(file_path='distances.csv'):
    distances = pd.read_csv(file_path)
    return distances

def ppmi_scores(corpus, window_size=20):
    finder = BigramCollocationFinder.from_words(corpus, window_size=window_size)
    bigram_measures = BigramAssocMeasures()
    pmi_scores = finder.score_ngrams(bigram_measures.pmi)
    pmi_dict = {}
    for pmi_score in pmi_scores:
        # From PMI to PPMI (more reliable)
        pmi_dict[pmi_score[0]] = max(pmi_score[1], 0)
    return pmi_dict

def train_cade_models(articles_by_decade, vector_dim, window_length, min_count,
                      stop_words_bool, bigrams_bool, trigrams_bool):
    """
    For all decades:
      1. Preprocess the text and save a CADE slice file (e.g., temp/cade_1980.txt).
      2. Concatenate all slices into a compass file (temp/compass.txt).
      3. Instantiate the CADE aligner (using the training parameters).
      4. Train the compass and then each slice (which yields aligned models).
      5. Save each model into the "model/" folder.
    """
    # Ensure the temporary folder and model folder exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    if not os.path.exists('model'):
        os.makedirs('model')
    
    compass_texts = []
    slice_files = {}
    
    # Process and save each decade’s text as a separate slice file.
    for index, row in articles_by_decade.iterrows():
        decade = row['Decade']
        content = row['Content']
        processed_data = preprocess_papers(content,
                                           stop_words=stop_words_bool,
                                           bigrams=bigrams_bool,
                                           trigrams=trigrams_bool)
        # Convert list of tokenized sentences to one text string (one sentence per line)
        slice_text = "\n".join([" ".join(sentence) for sentence in processed_data])
        slice_file = f"temp/cade_{decade}.txt"
        with open(slice_file, 'w', encoding='utf-8') as f:
            f.write(slice_text)
        slice_files[decade] = slice_file
        compass_texts.append(slice_text)
    
    # Create the compass file (concatenation of all slice texts)
    compass_file = "temp/compass.txt"
    with open(compass_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(compass_texts))
    
    st.write("Training the CADE compass...")
    # Initialize the CADE aligner – note that it accepts the same parameters as gensim's Word2Vec.
    aligner = CADE(size=vector_dim,
                   window=window_length,
                   min_count=min_count,
                   sg=1,
                   workers=multiprocessing.cpu_count())
    # Train the compass (if a compass file already exists and overwrite is False, it will reload the saved one)
    aligner.train_compass(compass_file, overwrite=True)
    st.write("Compass training completed.")
    
    # Train each slice (the returned model is already aligned)
    for decade, slice_file in slice_files.items():
        st.write(f"Training CADE slice for decade {decade}...")
        model = aligner.train_slice(slice_file, save=True)
        # Save the model to the desired location
        model_path = f"model/cade_{decade}.model"
        model.save(model_path)
        st.write(f"CADE model for decade {decade} saved to {model_path}.")

#####################
### USER INTERFACE
#####################

# Load the papers and group by decade (assumes a CSV with a 'Decade' column and a 'Content' column)
papers = pd.read_csv('data/papers.csv')
articles_by_decade = papers.groupby('Decade')['Content'].apply(lambda x: ' '.join(x)).reset_index()
articles_by_decade['Content'] = articles_by_decade['Content'].astype("string")

terms = load_terms('terms.json')
distances_df = pd.DataFrame(columns=['main term', 'decade', 'related term', 'distance', 'pmi'])

st.sidebar.header("Text preprocessing settings")
stop_words_bool = st.sidebar.checkbox("Remove stop words", True)
bigrams_bool = st.sidebar.checkbox("Include bigrams", True)
trigrams_bool = st.sidebar.checkbox("Include trigrams", True)

st.sidebar.header("Model settings")
min_count = st.sidebar.slider("Minimum count", 1, 10, value=2)
window_length = st.sidebar.slider("Window length", 2, 30, value=20)
vector_dim = st.sidebar.slider("Vectors dimension", 100, 300, value=100, step=50)

# When the user clicks the button, re-train all CADE models (compass + slices)
if st.sidebar.button("Re-train the models"):
    with st.spinner("Training the CADE models..."):
        train_cade_models(articles_by_decade, vector_dim, window_length, min_count,
                          stop_words_bool, bigrams_bool, trigrams_bool)

# For each decade, either load the pre-trained CADE model or (if missing) re-train all models.
for index, row in articles_by_decade.iterrows():
    decade = row['Decade']
    # Preprocess the text for PMI computation (even though the model was already trained on a preprocessed version)
    processed_data = preprocess_papers(row['Content'],
                                       stop_words=stop_words_bool,
                                       bigrams=bigrams_bool,
                                       trigrams=trigrams_bool)
    corpus = [word for sentence in processed_data for word in sentence]
    pmi_dict = ppmi_scores(corpus, window_size=50)
    
    model_path = f"model/cade_{decade}.model"
    if os.path.exists(model_path):
        with st.spinner(f'Loading the CADE model for decade {decade}...'):
            model = load_model(model_path)
    else:
        with st.spinner(f'Training the CADE model for decade {decade}...'):
            # (Since CADE trains all slices together, we re-run training for all decades if any is missing.)
            train_cade_models(articles_by_decade, vector_dim, window_length, min_count,
                              stop_words_bool, bigrams_bool, trigrams_bool)
            model = load_model(model_path)
    
    # Save PMI scores for this decade to a text file.
    with open(f'pmi_scores_{decade}.txt', 'w') as file:
        for key, value in pmi_dict.items():
            file.write(f'{key[0]} {key[1]} {value}\n')
    
    # For each term of interest, compute cosine similarity and collect distances.
    for main_term, related_terms in terms.items():
        for related_term in related_terms:
            cosine_distance = None
            pmi_score = None
            try:
                if main_term not in model.wv.vocab:
                    continue
                cosine_distance = model.wv.similarity(main_term, related_term)
                pmi_score = pmi_dict.get((main_term, related_term))
            except KeyError:
                cosine_distance = None
            distances_df = pd.concat([distances_df, pd.DataFrame({
                'main term': [main_term],
                'decade': [decade],
                'related term': [related_term],
                'distance': [cosine_distance],
                'pmi': [pmi_score]
            })])

# Save the distance data to a CSV file.
distances_df.to_csv('distances.csv', index=False)

# Display information and graphs
st.write('## Approach 1: Shift in cosine similarity between topic and related term')
with st.expander("How does the method work?"):
        st.write('''The idea is to project the words into a high-dimensional space where the words that share common contexts in the corpus are located close to each other. Using a distance metric such as cosine distance, we can measure the similarity between two words. These two words being the topic and the related term.
                    The implementation is the following:
- The dataset is divided into time slices (decades). All papers from a time slice are concatenated into one string.
- Stop words are removed, bigrams and trigrams are included. These options are configurable. As a result, a list of sentences is obtained.
- Train a “skip-gram with negative sampling (SGNS)” model for each time slice. Align the coordinates of the obtained models to ensure comparability. 
- In each time slice, calculate the cosine similarity between the topic and the related term.                 
                                  ''')
        st.info('''Note: The model SGNS (word2vec) is very sentitive to training settings with little data. To overcome our data limitations, we explicity opted for a lower min_count (count of a word in corpus to be considered by the model) and a longer window_length (the context length to be considered). These settings can be changed in the sidebar. We recommend to increase the min_count once more data is available.
                ''')
        st.info('''
                Another note: the quality of the results is subjective. There is no ground truth to compare the results with. Modifying the training settings will lead to different results.
                Number of epochs used for training: 25.
                ''')
with st.expander("How to interpret the graphs?"):
        st.write('''
For each topic in the "Terms of interest" tab, a graph is generated showing the cosine similarity between the topic and the related term across different decades. 
A higher cosine similarity indicates that the two words (in our scenario: topic and related term) are more similar in meaning (i.e., they share more contexts in the corpus). This doesn't mean that the topic and the related term co-occur more frequently, but rather that they are used in similar contexts! 
The graphs can be interpreted as follows:
- If the cosine similarity is close to 1, the topic and its related term are used in similar contexts and are likely to be related.
- If the cosine similarity is close to 0, the  topic and its related term are used in different contexts and are likely to be unrelated. 
                                  ''')
        st.info('Note: the comparison is between the topic and EACH related term across different time periods, and NOT between different related terms. The topic is in the title of the graph')
with st.spinner('Creating the graphs...'):
    distances_df = load_distances('distances.csv')
    main_terms = distances_df['main term'].unique()
    for main_term in main_terms:
        main_data = distances_df[distances_df['main term'] == main_term]
        chart = alt.Chart(main_data, title=main_term).mark_line(
            point=alt.OverlayMarkDef(filled=False, fill="white")
        ).encode(
            x=alt.X('decade:N', title='', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('distance', title='Cosine similarity'),
            color='related term',
            tooltip=['related term', 'distance']  
        )
        st.altair_chart(chart, theme='streamlit', use_container_width=True)


st.write('### Approach 2: Shift in co-occurence frequency between topic and related term')
distances_df = load_distances('distances.csv')
main_terms = distances_df['main term'].unique()
with st.expander("How does the PPMI (positive point-wise mutual information) work?"):
        st.write(''' 
        The method used is the positive point-wise mutual information (PPMI). It is a measure of association between two words. It is calculated as follows:
        ''')
        st.image('img/pmi_formula.png')
        st.write(''' 
        where 
        - P(topic term,related term) is the probability of the co-occurrence of 'topic term' and 'related term', and 
        - P(topic term) and P(related term) are the probabilities of 'topic term' and 'related term', respectively.
        
        In our context, the PPMI is calculated for each pair of <topic, related term> in each time slice (decade)
        ''')
        st.image('img/ppmi_formula.png')
        st.write(''' 
        - The graphs show how the PPMI score changes over time for the pair <topic, related term>. 
        - A higher PPMI score indicates that the 'topic' and 'related term' are more likely to co-occur in the same context.
        - The chosen context length is the same as for the cosine similarity calculation. You can change this in the sidebar.    
        
        This method is explainable and interpretable. It is based on the co-occurrence of words in the corpus. 
        The data needs to be large enough in order to have 'topic' and 'related term' in the same context window x times to be considered as a reliable measure. 
        ''')
        st.info('If the PPMI score is not present in the graph, it means that the pair <topic, related term> did not co-occur in the same context in the given time slice, which is bad news :(')



for main_term in main_terms:
    main_data = distances_df[distances_df['main term'] == main_term]


    ppmi = alt.Chart(main_data, title = main_term).mark_line(
        point=alt.OverlayMarkDef(filled=False, fill="white")
    ).encode(
        x=alt.X('decade:N', title='', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('pmi', title='PMI score'),
        color='related term',
        tooltip=['related term', 'pmi']  
    )

    st.altair_chart(ppmi, theme='streamlit', use_container_width=True)

