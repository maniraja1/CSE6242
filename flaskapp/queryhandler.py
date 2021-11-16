from transformers import AutoTokenizer, AutoModel
from csv import reader
import sklearn
import scipy
import pandas as pd
import os

tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

def getqueryembedding(query: str, abstract: str=''):
    #papers = [{'title': 'covid19', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]
    # concatenate title and abstract
    title_abs = [query + tokenizer.sep_token + abstract ]
    # preprocess the input
    inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings = result.last_hidden_state[:, 0, :]
    return embeddings


def getsimilarity(row,query_embedding):
    #print(row)
    vec1 = query_embedding.cpu().detach().numpy()
    vec2 = row.to_numpy()
    return 1 - scipy.spatial.distance.cosine(vec1, vec2)

def getrelateddocuments(query_title:str, query_abstract:str ):
    
    os.chdir(os.path.dirname(__file__))
    embedding = pd.read_csv('../data/embeddings.csv')

    query_embedding = getqueryembedding (query_title,query_abstract)

    embedding['similarity'] = embedding.apply(lambda row: getsimilarity(row[2:],query_embedding), axis=1)
    embedding = embedding.sort_values(by=['similarity'], ascending=False)
    return   embedding['ID'].head(10).to_numpy()


            

    



#