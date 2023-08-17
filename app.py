import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from flask import Flask, render_template, request

movies_data = pd.read_csv('movies.csv')

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

def get_movie_recommendations(movie_name, num_recommendations=30):
    find_close_match = difflib.get_close_matches(movie_name, movies_data['title'].tolist())
    if not find_close_match:
        return []  

    close_match = find_close_match[0]

    index_of_the_movie = movies_data[movies_data.title == close_match].index.values[0]

    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data.loc[index, 'title']
        recommendations.append(title_from_index)
        i += 1
        if i > num_recommendations:
            break
    return recommendations

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommended_movies = get_movie_recommendations(movie_name)
        if not recommended_movies:
            message = f"Sorry, no recommendations available for '{movie_name}'."
            return render_template('index.html', movie_name=movie_name, message=message)
        return render_template('index.html', movie_name=movie_name, recommended_movies=recommended_movies)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
