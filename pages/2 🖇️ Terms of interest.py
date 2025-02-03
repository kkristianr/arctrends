import streamlit as st
import json
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

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

@st.cache_resource
def load_model(path):
    model = Word2Vec.load(path)
    print("Model loaded")
    return model

st.write("## Terms generator")
st.write("Enter a term to generate a list of its related terms:")
term = st.text_input("Enter term:")
term = format_term(term)
if term is not None:
    try:
        model = load_model('model/cade_2016-2025.model')
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        st.stop()
    if st.button("Generate"):
        neighbours = list_neighbours(model, term, topn=100)
        neighbours = [word for word, _ in neighbours]
        st.write(f"Topic: {term}")
        st.write(f"Semantic similar neighbours: {', '.join(neighbours)}")


file_path = 'terms.json'
try:
    terms = load_terms(file_path)
    st.write("### Terms of interest")
    table_terms= {'Topics': list(terms.keys()),
                    'Related terms': [", ".join(terms[main_term]) for main_term in terms.keys()]}
    st.table(table_terms)
except FileNotFoundError:
    st.error("File not found.")

action = st.selectbox("Select action", ["Add topic", "Edit topic", "Delete topic", "Add related term", "Delete related term"])

if action == "Add topic":
    new_main_term = st.text_input("Enter new topic:")
    new_main_term = format_term(new_main_term)
    if st.button("Add"):
        terms[new_main_term] = []
        save_terms(terms, file_path)
        st.success("Topic added successfully.")

elif action == "Edittopic":
    main_term_to_edit = st.selectbox("Select topic to edit:", list(terms.keys()))
    edited_main_term = st.text_input("Enter edited topic:", value=main_term_to_edit)
    edited_main_term = format_term(edited_main_term)
    if st.button("Edit"):
        terms[edited_main_term] = terms.pop(main_term_to_edit)
        save_terms(terms, file_path)
        st.success("Topic edited successfully.")

elif action == "Delete topic":
    main_term_to_delete = st.selectbox("Select topic to delete:", list(terms.keys()))
    if st.button("Delete"):
        del terms[main_term_to_delete]
        save_terms(terms, file_path)
        st.success("Topic deleted successfully.")

elif action == "Add related term":
    main_term = st.selectbox("Select topic:", list(terms.keys()))
    new_related_term = st.text_input("Enter new related term:")
    new_related_term = format_term(new_related_term)
    if st.button("Add"):
        terms[main_term].append(new_related_term)
        save_terms(terms, file_path)
        st.success("Related term added successfully.")

elif action == "Delete related term":
    main_term = st.selectbox("Select topic:", list(terms.keys()))
    related_term_to_delete = st.selectbox("Select related term to delete:", terms[main_term])
    if st.button("Delete"):
        terms[main_term].remove(related_term_to_delete)
        save_terms(terms, file_path)
        st.success("Related term deleted successfully.")
