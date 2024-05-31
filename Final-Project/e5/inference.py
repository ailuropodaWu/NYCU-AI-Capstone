import lightning as pl
import os

def inference():
    data_path = "../dataset/document_data/historical"
    data = []
    for filename in os.listdir(data_path):
        if filename.endswith('.txt'):
            with open(os.path.join(data_path, filename)) as f:
                data.append(f.read())
    print(len(data))
    
if __name__ == "__main__":
    inference()