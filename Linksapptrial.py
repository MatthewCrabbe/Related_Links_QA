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
    return reading_time_min

def related_links(url):
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

    # Load the model configuration
    config = AutoConfig.from_pretrained('pytorch_model.bin')

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('pytorch_model.bin')

    # Load the model
    model = AutoModel.from_pretrained('pytorch_model.bin', config=config)
    
    # Load the dictionary from a file
    with open('dict_file.json', 'r') as file:
        articles = json.load(file)
        
    
    if url not in list(articles.keys()):   
        article_content = [x for x in articles.values()]

        # Function to get embeddings
        def get_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().numpy()

        # Assume `documents` is a list of your news articles 
        with open('arrays.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        new_url = url

        new_response = requests.get(new_url)
        new_soup = BeautifulSoup(new_response.text, 'html.parser')
        new_text = new_soup.find('meta', attrs={'name': 'description'}).get('content')
        if new_url not in list(articles.keys()):
            articles[new_url] = new_text

        new_embedding = get_embeddings(articles[new_url])

        # Compute similarity
        similarity_scores = [cosine_similarity(new_embedding, emb) for emb in embeddings]

        # Get top 5 similar documents
        top_5 = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:5]

        embeddings.append(new_embedding)

        # Save the list of arrays.
        with open('arrays.pkl', 'wb') as f:
            pickle.dump(embeddings, f)

        # Save the dictionary to a file
        with open('dict_file.json', 'w') as file:
            json.dump(articles, file)

        top_5_urls = []

        for i in top_5:
            top_5_urls.append([x for x in articles.keys()][i])
            top_5_urls = [x for x in top_5_urls if x != new_url]
            
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

        new_url = url

        new_response = requests.get(new_url)
        new_soup = BeautifulSoup(new_response.text, 'html.parser')
        new_text = new_soup.find('meta', attrs={'name': 'description'}).get('content')
        new_embedding = get_embeddings(articles[new_url])

        # Compute similarity
        similarity_scores = [cosine_similarity(new_embedding, emb) for emb in embeddings]

        # Get top 5 similar documents
        top_10 = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:10]

        top_5_urls = []

        for i in top_10:
            top_5_urls.append([x for x in articles.keys()][i])
            top_5_urls = [x for x in top_5_urls if x != new_url][:5]
            
    
    reading_time = estimate_reading_time(new_text)
    #print(f"Estimated reading time: {reading_time} minutes")
    

    return reading_time, top_5_urls

# Import Streamlit
import streamlit as st

# Title for the app
st.title('Related Article Links App')

# Take number input from the user
input_url = st.text_input('Enter article url:')

# Calculate the square of the number
time, links = related_links(input_url)

# Display the result
st.write(f'The estimated reading time is {time} and the URL links are {links}')
