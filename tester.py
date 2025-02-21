import pandas as pd
from recommender import load_data, build_tfidf_matrix, recommend_movies

dataset_path = "movies.csv"
df = load_data(dataset_path)
vectorizer, tfidf_matrix = build_tfidf_matrix(df)

#Test 1: Action-packed space 
query1 = "I love thrilling action movies set in space space space space with a comedic twist."
print(f"\nTest 1: {query1}")
print(recommend_movies(query1, df, vectorizer, tfidf_matrix))

#Test 2: Emotional family drama
query2 = "A heartwarming story about family and overcoming challenges heartwarming heartwarming."
print(f"\nTest 2: {query2}")
print(recommend_movies(query2, df, vectorizer, tfidf_matrix))

#Test 3: Superhero adventure
query3 = "Superheroes superheroes superheroes superheroes saving the world with exciting action sequences."
print(f"\nTest 3: {query3}")
print(recommend_movies(query3, df, vectorizer, tfidf_matrix))

#Test 4: Historical war
query4 = "A realistic and intense depiction of World War II battles."
print(f"\nTest 4: {query4}")
print(recommend_movies(query4, df, vectorizer, tfidf_matrix))

#Test 5: Nonsense 
query5 = "Haloopia Baloopia" 
print(f"\nTest 5: {query5}")
print(recommend_movies(query5, df, vectorizer, tfidf_matrix))
