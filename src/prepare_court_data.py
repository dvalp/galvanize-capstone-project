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
            json_itm = tf_files.extractfile(f_name)
            json_list.append(json.loads(json_itm.read().decode()))

    return json_list
