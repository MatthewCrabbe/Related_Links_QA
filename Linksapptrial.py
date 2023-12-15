import textstat
from transformers import AutoModel, AutoTokenizer, AutoConfig

def estimate_reading_time(text):
    num_words = len(text.split())
    flesch_score = textstat.flesch_reading_ease(text)

    # Adjust words-per-minute rate based on Flesch score
    if flesch_score >= 90.0:
        wpm = 250  # Very easy text
    elif flesch_score >= 80.0:
        wpm = 220  # Easy text
    elif flesch_score >= 70.0:
        wpm = 200  # Fairly easy text
    elif flesch_score >= 60.0:
        wpm = 180  # Standard text
    else:
        wpm = 150  # Fairly difficult, difficult, and very confusing text

    reading_time_min = num_words / wpm
    return round(reading_time_min, 2)

def related_links(title, article_text):
    #Import Modules
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    import json
    import pickle
    import numpy as np
    
    #save_directory = "/Users/david/Desktop/QA Media/Models"
    model_directory = './'

    # Load the model configuration
    config = AutoConfig.from_pretrained(model_directory)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # Load the model
    model = AutoModel.from_pretrained(model_directory, config=config)
    
    # Load the dictionary from a file
    with open('dict_file.json', 'r') as file:
        articles = json.load(file)
        
    
    if title not in list(articles.keys()):   
        article_content = [x for x in articles.values()]

        # Function to get embeddings
        def get_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().numpy()

        # Assume `documents` is a list of your news articles 
        with open('arrays.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        if title not in list(articles.keys()):
            articles[title] = article_text

        new_embedding = get_embeddings(articles[title])

        # Compute similarity
        similarity_scores = [cosine_similarity(new_embedding, emb) for emb in embeddings]

        # Get top 5 similar documents
        top_15 = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:15]
        
        embeddings.append(new_embedding)

        # Save the list of arrays.
        with open('arrays.pkl', 'wb') as f:
            pickle.dump(embeddings, f)

        # Save the dictionary to a file
        with open('dict_file.json', 'w') as file:
            json.dump(articles, file)

        top_10_titles = []

        for i in top_15:
            top_10_titles.append([x for x in articles.keys()][i])
            top_10_titles = [x for x in top_10_titles if x != title][:5]
            
    else:
        article_content = [x for x in articles.values()]

        # Function to get embeddings
        def get_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().numpy()

        # Assume `documents` is a list of your news articles 
        with open('arrays.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        new_embedding = get_embeddings(articles[title])

        # Compute similarity
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


