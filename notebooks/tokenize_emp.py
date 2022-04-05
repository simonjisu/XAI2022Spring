import spacy
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# class Tokenizer():
#     def __init__(self, model='en_core_web_sm', only_token=False):
#         if only_token:
#             self.nlp = spacy.load(model, disable=['parser', 'tagger', 'ner'])
#         else:
#             self.nlp = spacy.load(model)

#     def tokenize(self, x):
#         return list(map(lambda t: t.text, self.nlp(x)))

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
main_path = Path().absolute().parent
data_path = main_path / 'data' / 'p2p' / 'lending_club' / 'processed'

df = pd.read_csv( data_path / 'accepted.csv')
df['emp_title'].fillna('unknown', inplace=True)
df['emp_title'] = df['emp_title'].str.lower()  # Unify into lower cases
emp_title = df['emp_title'].to_list()
with (data_path / 'emp_title.txt').open('w', encoding='utf-8') as file:
    docs = nlp.pipe(emp_title, n_process=4, batch_size=2000)
    for x in tqdm(docs, total=len(list(docs))):
        print('\t'.join([t.text for t in x]), file=file)
