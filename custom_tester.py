from recommender import load_data, build_tfidf_matrix, recommend_movies

df = load_data("movies.csv")
vectorizer, tfidf_matrix = build_tfidf_matrix(df)
query = "I love thrilling action movies set in space with a comedic twist."
print(recommend_movies(query, df, vectorizer, tfidf_matrix))