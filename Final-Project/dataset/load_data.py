from datasets import load_dataset
import pandas as pd

data_split = ["train", "validation", "test"]

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
    for name in data_list:
        data = load_data(name)
        for split in data_split:
            if data[split] is None:
                continue
            for _, row in data[split].iterrows():
                df.append([row["sentence1"], row["sentence2"], row["score"]])
    df = pd.DataFrame(df, columns=["sentence1", "sentence2", "score"])
    df.drop_duplicates(inplace=True, ignore_index=True)
    df.to_json("./data.json", orient="records")
    print(df)