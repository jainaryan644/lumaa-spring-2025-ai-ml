import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(filepath):
    #load dataset
    df = pd.read_csv(filepath)

    #print columns for dataset
    print("Columns in dataset:", df.columns)
    print(df.head())

    return df

def build_tfidf_matrix(df):
    #text to tfidf matrix conversion
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Description'].fillna(''))
    return vectorizer, tfidf_matrix


if __name__ == "__main__":
    path = "movies.csv"
    df = load_data(path)
    vectorizer, tfidf_matrix = build_tfidf_matrix(df)
