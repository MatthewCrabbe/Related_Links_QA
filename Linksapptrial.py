import textstat
from transformers import AutoModel, AutoTokenizer, AutoConfig

def related_links(title, article_text):
    # Import Modules (unchanged)
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    import json
    import pickle
    import numpy as np
    
    model_directory = './'

    # Load the model configuration (unchanged)
    config = AutoConfig.from_pretrained(model_directory)

    # Load the tokenizer (unchanged)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # Load the model (unchanged)
    model = AutoModel.from_pretrained(model_directory, config=config)
    
    # Load the dictionary from a file (unchanged)
    with open('dict_file.json', 'r') as file:
        articles = json.load(file)
    
    if title not in list(articles.keys()):   
        article_content = [x for x in articles.values()]

        # Function to get embeddings (unchanged)
        def get_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().numpy()

        # Assume `documents` is a list of your news articles 
        with open('arrays.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        articles[title] = article_text  # Add the new article to the dictionary

        new_embedding = get_embeddings(articles[title])

        # Compute similarity (unchanged)
        similarity_scores = [cosine_similarity(new_embedding, emb) for emb in embeddings]

        # Get top 5 similar documents (unchanged)
        top_15 = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:15]
        
        embeddings.append(new_embedding)

        # Save the list of arrays (unchanged)
        with open('arrays.pkl', 'wb') as f:
            pickle.dump(embeddings, f)

        # Save the dictionary to a file (unchanged)
        with open('dict_file.json', 'w') as file:
            json.dump(articles, file)

        top_10_titles = []

        for i in top_15:
            top_10_titles.append([x for x in articles.keys()][i])
            top_10_titles = [x for x in top_10_titles if x != title][:5]
    else:
        # If the title is already in the dictionary, simply get related articles
        article_content = [x for x in articles.values()]

        # Function to get embeddings (unchanged)
        def get_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().numpy()

        # Assume `documents` is a list of your news articles 
        with open('arrays.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        new_embedding = get_embeddings(articles[title])

        # Compute similarity (unchanged)
        similarity_scores = [cosine_similarity(new_embedding, emb) for emb in embeddings]

        top_15 = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:15]

        top_10_titles = []

        for i in top_15:
            top_10_titles.append([x for x in articles.keys()][i])
            top_10_titles = [x for x in top_10_titles if x != title][:5]

    reading_time = estimate_reading_time(article_text)

    return reading_time, top_10_titles
    
# Import Streamlit
import streamlit as st

# Title for the app
st.title('Related Article Links App')

# Take title input from the user
input_title = st.text_input('Enter article title:')

#Take text input from the user
input_article_content = st.text_area('Enter article body:', height=300)

button_pressed = st.button("Submit")

if button_pressed:
    time, links = related_links(input_title, str(input_article_content))
    # Display the result
    st.write(f'The estimated reading time is {time} minutes and the related titles are:')
    for x in links:
        parts = x.split("|")
        result = parts[0].strip()
        st.write(result)
else:
    st.write('Please enter title and body')

