import pandas as pd
import tarfile
import json
from bs4 import BeautifulSoup

# Import the CiteGeist file into a dataframe
# CiteGeist uses tfidf as part of its ranking and isn't currently helpful to my project
# df = pd.read_table('data/citegeist', header=None, names=['rating'], delimiter='=', index_col=0)

# Extract the json files from tar.gz files and return a DataFrame
def create_df_from_tar(files_tar):
    json_list = []

    with tarfile.open(files_tar, mode='r:gz') as tf_files:
        for f_name in tf_files.getnames():

            # import the text and create a dict
            text_itm = tf_files.extractfile(f_name)
            json_itm = json.loads(text_itm.read().decode())

            # parse the html out of each record as it is imported
            if json_itm.get('html_columbia'):
                json_itm['html_columbia'].replace('<blockquote>', '"').replace('</blockquote>', '"')
                json_itm['parsed_columbia'] = BeautifulSoup(json_itm['html_columbia'], 'lxml').text
            if json_itm.get('html_lawbox'):
                json_itm['html_lawbox'].replace('<blockquote>', '"').replace('</blockquote>', '"')
                json_itm['parsed_lawbox'] = BeautifulSoup(json_itm.get('html_lawbox'), 'lxml').text

            json_list.append(json_itm)

    return json_list
