from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle

df = pd.read_csv('movie.csv')

similarity_matrix = pickle.load(open('similarity_matrix.pkl', 'rb'))

cv = CountVectorizer(stop_words='english')


app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form.get('movie_name')
    movie_list = recommender(movie_name)
    return render_template('index.html', movie_list=movie_list)

def recommender(movie_name):

    index_pos = df[df['title'] == movie_name].index[0]

    recommended_movie_index = sorted(list(enumerate(similarity_matrix[index_pos])), reverse=True, key=lambda x: x[1])[1:11]

    movie_list=[]

    for i in recommended_movie_index:
        movie_list.append(df.iloc[i[0]].title)

    return movie_list

if __name__ == "__main__":
    app.run(debug=True)
