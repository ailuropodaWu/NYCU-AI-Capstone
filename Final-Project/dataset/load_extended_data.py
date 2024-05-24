import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

data_split = ["train", "validation", "test"]

def calculate_tfidf_cosine(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

if __name__ == '__main__':
    df = pd.read_json("./extended_data.json")
    tfidf_cos = []
    
    df.drop_duplicates(inplace=True, ignore_index=True)
    for _, row in df.iterrows():
        cos_sim = calculate_tfidf_cosine(row['paragraph1'], row['paragraph2'])
        tfidf_cos.append(cos_sim)
    df['tfidf-cos'] = tfidf_cos
    df.to_json("./paragraph_data.json", orient="records")
    print(df)