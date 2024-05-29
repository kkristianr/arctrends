import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pymupdf
from multi_column import column_boxes

from wordcloud import WordCloud

import streamlit as st
from streamlit import components
from streamlit import session_state as ss

papers = pd.read_csv('data/sust.csv')
stopwords = ["et", "figure", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def binning_decades(papers):
    bins = [1975, 1985, 1995, 2005, 2015, 2025, 2035, 2045]
    labels = ['1976-1985', '1986-1995', '1996-2005', '2006-2015', '2016-2025', '2026-2035', '2036-2045']
    papers['Decade'] = pd.cut(papers['Year'], bins=bins, labels=labels)
    return papers

#papers = binning_decades(papers)



articles_by_decade = papers.groupby('Decade')['Content'].apply(lambda x: ' '.join(x)).reset_index()

### USERFACE
 
dataset_tab, pdf_reader, clouds_tab, about_tab  = st.tabs(["Dataset", "Paper reader", "Word clouds", "About"])

with dataset_tab:
    st.write("## About the dataset")
    count = len(papers)
    st.write("Description about the source and the method of data collection are coming soon")
    st.write("The method is based on calculating probabilities of words being in a certain context. In order for it to work reasonably well, we need a large number of manually selected papers (of high quality)")
    st.write(f"So far, number of papers included: {count}")
    
    table_data = {'DOI': papers['DOI'],
                  'Title': papers['Title'], 
                  'Year': papers['Year']}
    st.table(table_data)

    st.write("## Dataset stastics")
    papers['Year'] = papers['Year'].astype(int)

    papers_by_decade = papers.groupby('Decade').size().reset_index(name='Papers')
    st.write("Number of papers by decade")
    st.bar_chart(papers_by_decade.set_index('Decade'))

with pdf_reader: 
    import fitz
    st.write("## 1. Add a paper through PDF file")
    new_doi = st.text_input("DOI:", key='doi')
    paper_year = st.text_input("Publication year:", key='year')
    new_title = st.text_input("Paper title:", key='title')
    with st.popover("Adjust the PDF reader settings"):
        st.info("The pdf reader is based on the pymupdf library. As default, two-columns layout is assumed. More information [here](https://artifex.com/blog/extract-text-from-a-multi-column-document-using-pymupdf-inpython)")
        footer_margin = st.slider("Footer margin", 0, 100, 40)
        header_margin = st.slider("Header margin", 0, 100, 60)
        no_image_text = st.checkbox("Exclude image text", value=True)
    
    ## if doi already exists, do not add the paper
    if new_doi in papers['DOI'].values:
        st.error("This paper already exists in the dataset")
        st.stop()
    pdf_file = st.file_uploader("Upload PDF file", type="pdf", key='pdf')
    if pdf_file is not None:
        pdf_bytes = pdf_file.getvalue()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            bboxes = column_boxes(page, footer_margin=footer_margin, header_margin=header_margin, no_image_text=no_image_text)
            for rect in bboxes:
                text += page.get_text("text", clip=rect)

        updated_text = st.text_area("Extractred content of paper", text, height=850)
        
        st.info('Go through the text, remove unrelevant content such as first page content, references, acknowledgements and eventually footer and header content', icon="ℹ️")
        confirm_content = st.button("Confirm and add the paper")
        if confirm_content:
            new_paper = pd.DataFrame(columns=['DOI', 'Title', 'Year', 'Content'])
            new_paper['DOI'] = [new_doi]
            new_paper['Title'] = [new_title]
            new_paper['Year'] = [paper_year]
            new_paper['Decade'] = pd.cut([paper_year], bins=[1975, 1985, 1995, 2005, 2015, 2025, 2035, 2045], labels=['1976-1985', '1986-1995', '1996-2005', '2006-2015', '2016-2025', '2026-2035', '2036-2045'])
            new_paper['Content'] = [updated_text]
            papers = pd.concat([papers, new_paper])
            ##papers = papers.drop_duplicates(subset=['DOI'])
            papers.to_csv('data/sust.csv', index=False)
            st.success("New paper successfully added")
        doc.close()
        
    st.write("## 2. Upload CSV file with extracted text")
    st.write("To add new papers, please upload a CSV file with the following columns: DOI, Title, Year, Content where the column Content contains the extracted text of the paper.")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        new_papers = pd.read_csv(uploaded_file)
        if set(new_papers.columns) != set(['DOI', 'Title', 'Year', 'Content']):
            st.write("Please make sure the uploaded file contains the following columns: DOI, Title, Year, Content")
            st.stop()

        papers = pd.concat([papers, new_papers])
        papers = papers.drop_duplicates(subset=['DOI'])
        papers = papers.binning_decades(papers)

        st.write("New papers added")
        count = len(papers)
        st.write(f"Total number of papers included: {count}")

with about_tab: 
    st.write("## About the method")
    st.write("It is a custom implementation based on computing word2vec embeddings for some specific words of interest such as SUSTAINABILITY or PATIENT PRIVACY. By looking at the neighborhood of these words in the vector space and comparing the distance across different decades, we try to identify shifts in the healthcare architectural design by visualizing the changes in the distance between neighbours and the words of interest.")
    st.write("## About the project")

with clouds_tab: 

    st.write("Word clouds for every decade")
    st.write("The word clouds show the most frequent words in the papers for each year.")


    for decade, dataset in zip(articles_by_decade['Decade'], articles_by_decade['Content']):
        tokens_count = len(dataset.split())
        st.write(f"Decade: {decade}, Number of tokens: {tokens_count}")

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
        plt.title(decade)
        st.pyplot(plt)