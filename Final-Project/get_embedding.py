import sys
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--not_use_saved", action="store_true", default=False)
args = parser.parse_args()
model = args.model
use_saved = not args.not_use_saved

sys.path.append(f"./{model}/")
from embedding import get_embedding

embedding_dict = get_embedding(os.path.join(model, "embedding_dict.json"), use_saved=use_saved)