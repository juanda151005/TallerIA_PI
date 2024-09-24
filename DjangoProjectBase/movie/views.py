from django.shortcuts import render
from django.http import HttpResponse

from .models import Movie
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import os

_ = load_dotenv('../api_keys.env')
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get('openai_apikey'),
)

# Cargar las descripciones de las películas con embeddings
with open('../movie_descriptions_embeddings.json', 'r') as file:
    movies_with_embeddings = json.load(file)

# Función para obtener el embedding del texto
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Función para calcular la similitud de coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def home(request):
    searchTerm = request.GET.get('searchMovie')  # Obtener el término de búsqueda
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm': searchTerm, 'movies': movies})


def about(request):
    return render(request, 'about.html')


def signup(request):
    email = request.GET.get('email') 
    return render(request, 'signup.html', {'email': email})


def recommend_movie(request):
    search_term = request.GET.get('searchMovie', '')  # Obtener el prompt del formulario
    recommended_movies = []

    if search_term:
        embedding_prompt = get_embedding(search_term)  # Generar el embedding del prompt

        # Calcular las similitudes entre el prompt y las películas
        similarities = []
        for movie in movies_with_embeddings:
            similarities.append(cosine_similarity(embedding_prompt, movie['embedding']))

        # Ordenar las películas por similitud
        sorted_indices = np.argsort(similarities)[::-1]

        for idx in sorted_indices[:1]:
            recommended_movies.append(movies_with_embeddings[idx])

    return render(request, 'recommend.html', {'searchTerm': search_term, 'movies': recommended_movies})


def statistics_view0(request):
    matplotlib.use('Agg')
    all_movies = Movie.objects.all()

    movie_counts_by_year = {}
    for movie in all_movies:
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    bar_width = 0.5
    bar_positions = range(len(movie_counts_by_year))
    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center')

    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=90)

    plt.subplots_adjust(bottom=0.3)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'statistics.html', {'graphic': graphic})


def statistics_view(request):
    matplotlib.use('Agg')
    all_movies = Movie.objects.all()

    movie_counts_by_year = {}
    for movie in all_movies:
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    year_graphic = generate_bar_chart(movie_counts_by_year, 'Year', 'Number of movies')

    movie_counts_by_genre = {}
    for movie in all_movies:
        genres = movie.genre.split(',')[0].strip() if movie.genre else "None"
        if genres in movie_counts_by_genre:
            movie_counts_by_genre[genres] += 1
        else:
            movie_counts_by_genre[genres] = 1

    genre_graphic = generate_bar_chart(movie_counts_by_genre, 'Genre', 'Number of movies')

    return render(request, 'statistics.html', {'year_graphic': year_graphic, 'genre_graphic': genre_graphic})


def generate_bar_chart(data, xlabel, ylabel):
    keys = [str(key) for key in data.keys()]
    plt.bar(keys, data.values())
    plt.title('Movies Distribution')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic
