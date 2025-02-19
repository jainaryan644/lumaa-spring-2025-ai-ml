# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

## Dataset
The dataset used is **IMDb Top 250 Movies**.

- **Source:** IMDb (pre-provided in the challenge).  
- **File:** `movies.csv`  
- **Columns Used:** `Name`, `Description`, `Stars`, `Director`  
- **Loading:** The script automatically loads `movies.csv`. No manual steps required.

---

## **Setup Instructions**
### **Python Version**
- Recommended: **3.8+**

### **Install Dependencies**
Create and activate a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

### **Running the System**
To run the recommendation system, use:

```sh
python test_recommender.py

This script automatically runs five test cases and displays recommended movies for each query.

However, If you want to test a custom test case use this Python script:

from recommender import load_data, build_tfidf_matrix, recommend_movies

df = load_data("movies.csv")
vectorizer, tfidf_matrix = build_tfidf_matrix(df)
query = "I love thrilling action movies set in space with a comedic twist."
print(recommend_movies(query, df, vectorizer, tfidf_matrix))

### **Example Query and Output:**

Test 1: I love thrilling action movies set in space space space space with a comedic twist.
                Name                                        Description
245   The Iron Giant  A young boy befriends a giant robot from outer...
178     Blade Runner  A blade runner must pursue and terminate four ...
57            WALL·E  In the distant future, a small waste-collectin...
227  The Incredibles  While trying to lead a quiet suburban life, a ...
119  Bicycle Thieves  In post-war Italy, a working-class man's bicyc...


### **Known Limitations**
The model relies only on text similarity, so it may sometimes match movies with related words but different themes.
Genre filtering is not included since the dataset lacks genre information.
A more advanced approach would use word embeddings (e.g., Word2Vec, BERT) to capture deeper meanings.
