import pandas as pd
import tarfile

# Import the CiteGeist file into a dataframe
# CiteGeist uses tfidf as part of its ranking and isn't currently helpful to my project
# df = pd.read_table('data/citegeist', header=None, names=['rating'], delimiter='=', index_col=0)

# Extract the json files from opinions_wash.tar.gz
tf_opinions = tarfile.open('data/opinions_wash.tar.gz', mode='r:gz')
opinion_names = tf_opinions.getnames()

opinions_list = []
for name in opinion_names:
    opinion = tf_opinions.extractfile(name)
    opinions_list.append(json.loads(opinion.read().decode().replace('\n', '')))

