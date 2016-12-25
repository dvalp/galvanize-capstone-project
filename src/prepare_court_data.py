import pandas as pd
import numpy as np
import tarfile
import json
from bs4 import BeautifulSoup

# Import the CiteGeist file into a dataframe
# CiteGeist uses tfidf as part of its ranking and isn't currently helpful to my project
# df = pd.read_table('data/citegeist', header=None, names=['rating'], delimiter='=', index_col=0)


def create_list_from_tar(files_tar, length=None):
    '''
    Extract the json files from tar.gz files and return a list of selected data
    '''
    json_list = []

    with tarfile.open(files_tar, mode='r:gz') as tf_files:
        filenames = tf_files.getnames()[:length]
        for f_name in filenames:

            # import the text and create a dict
            text_itm = tf_files.extractfile(f_name)
            json_itm = json.loads(text_itm.read().decode())

            # parse the html out of each record as it is imported
            if json_itm.get('html_columbia'):
                json_itm['html_columbia'] = json_itm['html_columbia'].replace('<blockquote>', '"').replace('</blockquote>', '"')
                json_itm['parsed_text'] = BeautifulSoup(json_itm['html_columbia'], 'lxml').text
            elif json_itm.get('html_lawbox'):
                json_itm['html_lawbox'] = json_itm['html_lawbox'].replace('<blockquote>', '"').replace('</blockquote>', '"')
                json_itm['parsed_text'] = BeautifulSoup(json_itm.get('html_lawbox'), 'lxml').text
            elif json_itm.get('html_with_citations'):
                json_itm['html_with_citations'] = json_itm['html_with_citations'].replace('<blockquote>', '"') \
                        .replace('</blockquote>', '"')
                json_itm['html_with_citations'] = BeautifulSoup(json_itm.get('html_with_citations'), 'lxml').text
            elif json_itm.get('plain_text'):
                json_itm['parsed_text'] = json_itm.get('plain_text')

            json_itm['cluster_id'] = int(json_itm['cluster'].split('/')[-2])
            json_itm['resource_id'] = int(json_itm['resource_uri'].split('/')[-2])
            citations = []
            for cite in json_itm['opinions_cited']:
                citations.append(int(cite.split('/')[-2]))
            json_itm['opinions_cited'] = citations

            json_list.append(json_itm)

    return json_list

def reverse_stem(resource_id, opinion_df, opinion_cv_model, df_stems):
    '''
    Take the stemmed words in a document and return the possible words (from all documents) that could 
    could have been used to create the stem. This doesn't (yet) take into account whether the specific 
    words actually exist in the current document.
    '''
    row = opinion_df.filter(opinion_df.resource_id == resource_id).first()
    term_stems = np.array(opinion_cv_model.vocabulary)[row['token_idf'].indices[np.argsort(row['token_idf'].values)]][:-11:-1]
    word_lists = []
    for stem in term_stems:
        word_lists.append(df_stems.select('terms').filter(df_stems.stem == stem).first()[0])
    return word_lists

def create_df(tar_file, length=None):
    '''
    Use Spark to import files from a tarfile and store the json information directly into a Spark dataframe. 
    
    This should replace the previous function: create_list_from_tar()
    
    First, loop through the TarInfo objects to get filenames rather than loading the whole list of names at once. 
    Second, load each file one at a time to parallelize them, rather than loading all the files into memory.
    Finally, read the json and separate the fields into columns if possible.
    '''
    pass
    
