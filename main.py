from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar el archivo Parquet
df = pd.read_parquet('Dataset/movies.parquet')  # Reemplaza con la ruta correcta al archivo Parquet


# Crear una instancia de FastAPI
app = FastAPI()

# Asegúrate de que las columnas no tengan valores nulos para este caso específico
df['overview'] = df['overview'].fillna('')  # Rellenar descripciones nulas con cadenas vacías

# Inicializar el vectorizador TF-IDF
tfidf = TfidfVectorizer(stop_words='english')

# Generar la matriz TF-IDF basada en la columna 'overview'
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Calcular la matriz de similitud de coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Crear un índice de título para acceso rápido
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    # Convertir el mes ingresado a minúsculas para asegurarnos de que la comparación sea insensible a mayúsculas/minúsculas
    mes = mes.lower()

    # Filtrar el DataFrame para contar las películas estrenadas en el mes especificado
    cantidad = df[df['mes'].str.lower() == mes].shape[0]

    # Verificar si se encontraron películas
    if cantidad == 0:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas estrenadas en el mes '{mes}'.")

    # Devolver la cantidad de películas
    return {
        "mes": mes,
        "cantidad_filmaciones": cantidad
    }


@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    # Convertir el día ingresado a minúsculas para asegurar que la comparación sea insensible a mayúsculas/minúsculas
    dia = dia.lower()

    # Filtrar el DataFrame para contar las películas estrenadas en el día especificado
    cantidad = df[df['dia'].str.lower() == dia].shape[0]

    # Verificar si se encontraron películas
    if cantidad == 0:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas estrenadas en el día '{dia}'.")

    # Devolver la cantidad de películas
    return {
        "dia": dia,
        "cantidad_filmaciones": cantidad
    }



@app.get("/score_titulo/{titulo_de_la_filmacion}")
def score_titulo(titulo_de_la_filmacion: str):
    # Filtrar el DataFrame para encontrar la filmación por título
    filmacion = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]

    # Verificar si se encontró alguna filmación
    if filmacion.empty:
        raise HTTPException(status_code=404, detail="Filmación no encontrada")

    # Extraer la información necesaria
    titulo = filmacion.iloc[0]['title']
    # Obtener el año de la fecha de estreno
    anio_estreno = filmacion.iloc[0]['release_date'].year
    score = filmacion.iloc[0]['vote_average']

    # Devolver el resultado
    return {"titulo": titulo, "anio_estreno": anio_estreno, "score": score}


@app.get("/votos_titulo/{titulo_de_la_filmacion}")
def votos_titulo(titulo_de_la_filmacion: str):
    # Filtrar el DataFrame para encontrar la película con el título dado
    pelicula = df[df['title'].str.lower() == titulo_de_la_filmacion.lower()]

    # Verificar si la película fue encontrada
    if pelicula.empty:
        raise HTTPException(status_code=404, detail=f"Película '{titulo_de_la_filmacion}' no encontrada en el dataset.")

    # Obtener los detalles de la película
    votos = pelicula['vote_count'].values[0]
    promedio_votos = pelicula['vote_average'].values[0]
    titulo = pelicula['title'].values[0]

    # Verificar si la cantidad de votos es al menos 2000
    if votos < 2000:
        raise HTTPException(status_code=400, detail=f"La película '{titulo}' no cumple con la cantidad mínima de 2000 valoraciones.")

    # Devolver el resultado
    return {
        "titulo": titulo,
        "cantidad_votos": votos,
        "promedio_votos": promedio_votos
    }


@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    # Normalizar el nombre del actor para búsqueda insensible a mayúsculas
    nombre_actor = nombre_actor.lower()

    # Filtrar el DataFrame para encontrar las películas en las que el actor ha participado
    def actor_in_cast(cast_array):
        # Asegurarse de que cast_array sea un numpy.ndarray y manejar errores
        if isinstance(cast_array, np.ndarray):
            return nombre_actor in [actor.lower() for actor in cast_array]
        return False

    # Usar la función 'apply' para filtrar el DataFrame
    filmaciones_actor = df[df['cast'].apply(actor_in_cast)]

    # Verificar si se encontró alguna participación
    if filmaciones_actor.empty:
        raise HTTPException(status_code=404, detail=f"Actor '{nombre_actor}' no encontrado o sin participaciones en el dataset.")

    # Calcular el revenue total, cantidad de películas, y promedio de revenue
    revenue_total = filmaciones_actor['revenue'].sum()
    cantidad_peliculas = len(filmaciones_actor)
    promedio_revenue = revenue_total / cantidad_peliculas if cantidad_peliculas > 0 else 0

    # Devolver el resultado
    return {
        "actor": nombre_actor,
        "cantidad_peliculas": cantidad_peliculas,
        "revenue_total": revenue_total,
        "promedio_revenue": promedio_revenue
    }

    

@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    try:
        # Filtrar el DataFrame para encontrar las películas dirigidas por el director
        def director_in_list(director_list):
            # Verifica si director_list es un array o lista válida
            if isinstance(director_list, np.ndarray):
                director_list = director_list.tolist()
            if not isinstance(director_list, list) or not director_list:
                return False
            return nombre_director.lower() in [director.lower() for director in director_list]

        # Aplicar la función para filtrar el DataFrame
        peliculas_director = df[df['director'].apply(director_in_list)]

        # Verificar si se encontró alguna película
        if peliculas_director.empty:
            raise HTTPException(status_code=404, detail=f"Director '{nombre_director}' no encontrado o sin películas en el dataset.")

        # Calcular el éxito del director y preparar los detalles de las películas
        detalles_peliculas = []
        for _, pelicula in peliculas_director.iterrows():
            titulo = pelicula['title']
            fecha_lanzamiento = pelicula['release_date']
            costo = pelicula['budget']
            ganancia = pelicula['revenue']
            retorno_individual = ganancia - costo  # Ganancia menos el presupuesto

            detalles_peliculas.append({
                "titulo": titulo,
                "fecha_lanzamiento": fecha_lanzamiento,
                "retorno_individual": retorno_individual,
                "costo": costo,
                "ganancia": ganancia
            })

        # Calcular el retorno total del director
        retorno_total = sum(p['retorno_individual'] for p in detalles_peliculas)

        # Devolver el resultado
        return {
            "director": nombre_director,
            "retorno_total": retorno_total,
            "peliculas": detalles_peliculas
        }

    except Exception as e:
        # Capturar errores inesperados y devolver un mensaje de error
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    # Verificar si el título existe en el índice
    if titulo not in indices:
        raise HTTPException(status_code=404, detail=f"La película '{titulo}' no se encuentra en el dataset.")

    # Obtener el índice de la película que coincide con el título
    idx = indices[titulo]

    # Obtener las similitudes de coseno de esa película con todas las demás
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar las películas basadas en la similitud de coseno (de mayor a menor)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de las 5 películas más similares, ignorando la primera (que es la misma película)
    sim_indices = [i[0] for i in sim_scores[1:6]]

    # Devolver los títulos de las 5 películas más similares
    recomendaciones = df['title'].iloc[sim_indices].tolist()

    return {
        "titulo": titulo,
        "recomendaciones": recomendaciones
    }