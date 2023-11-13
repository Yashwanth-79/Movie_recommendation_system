import streamlit as st

# Set the background image URL
background_image_url = "https://mcdn.wallpapersafari.com/medium/79/14/VpRCdM.jpg"
# Apply background image using CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movies data
movies_data = pd.read_csv('movies.csv')

# Data preprocessing
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director', 'original_title']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna(' ')

combined_feature = movies_data.genres + ' ' + movies_data.keywords + ' ' + movies_data.tagline + ' ' + movies_data.cast + ' ' + movies_data.director + ' ' + movies_data.original_title

# Convert text data into feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_feature)

# Cosine Similarity
similarity = cosine_similarity(feature_vectors)

# Streamlit App
st.title("Movie Recommender System")

# User input
movie_name = st.text_input("Enter your interested English movie name:")

if st.button("Get Recommendations"):
    # Find close matches
    find_close_match = difflib.get_close_matches(movie_name, movies_data['title'].tolist())
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        
        # Get the list of similar movies
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[:10]

        # Display suggested movies
        st.subheader('Suggested movies for you:')
        for i, movie in enumerate(sorted_similar_movies):
            if i <11:
                index = movie[0]
                title_from_index = movies_data[movies_data.index == index]['title'].values[0]
                st.write(f"{i + 1}. {title_from_index}")
    else:
        st.warning("No close match found. Please enter a valid movie name.")
