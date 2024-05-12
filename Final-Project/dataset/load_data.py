from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

data_split = ["train", "validation", "test"]

def calculate_tfidf_cosine(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def load_data(name:str):
    dataset = load_dataset(name)
    dataset.set_format("pandas")
    data = {}
    for split in data_split:
        data[split] = dataset[split][:] if split in dataset.keys() else None
        
    return data
    
if __name__ == '__main__':
    data_list = ['mteb/stsbenchmark-sts', 'mteb/sts12-sts', 'mteb/sts13-sts', 'mteb/sts14-sts', 'mteb/sts15-sts']
    df = []
    tfidf_cos = []
    for name in data_list:
        data = load_data(name)
        for split in data_split:
            if data[split] is None:
                continue
            for _, row in data[split].iterrows():
                df.append([row["sentence1"], row["sentence2"], row["score"]])
    df = pd.DataFrame(df, columns=["sentence1", "sentence2", "score"])
    df.drop_duplicates(inplace=True, ignore_index=True)
    for _, row in df.iterrows():
        cos_sim = calculate_tfidf_cosine(row['sentence1'], row['sentence2'])
        tfidf_cos.append(cos_sim)
    df['tfidf-cos'] = tfidf_cos
    df.to_json("./data.json", orient="records")
    print(df)