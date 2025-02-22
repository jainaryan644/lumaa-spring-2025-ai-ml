import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(filepath):
    #load dataset
    df = pd.read_csv(filepath, encoding="ISO-8859-1")

    #print columns for dataset
    #print("Columns in dataset:", df.columns)
    #print(df.head())

    df['combined_text'] = (
        df['Description'].fillna('') + ' ' +
        df['Stars'].fillna('') + ' ' +
        df['Director'].fillna('') 
    )

    return df

#text to tfidf matrix conversion
def build_tfidf_matrix(df):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2)) #bigrams for better context
    tfidf_matrix = vectorizer.fit_transform(df['Description'].fillna(''))
    return vectorizer, tfidf_matrix

#recommend top movies based on cosine similarity 
def recommend_movies(user_query, df, vectorizer, tfidf_matrix, top_n=5):
    query_vec = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]

    if similarity_scores[top_indices[0]] < 0.05:
        print("Your query may not have strong matches")

    recommended_df = df.iloc[top_indices][['Name']]

    #retrieve descriptions separately
    recommended_df['Description'] = recommended_df['Name'].apply(
        lambda name: df.loc[df['Name'] == name, 'Description'].values[0]
    )
    return recommended_df

#main
if __name__ == "__main__":
    path = "movies.csv"
    df = load_data(path)
    vectorizer, tfidf_matrix = build_tfidf_matrix(df)
