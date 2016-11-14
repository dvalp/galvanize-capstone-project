from src import prepare_court_data
import pyspark as ps
from bs4 import BeautifulSoup
import pandas as pd


opinion_lst = prepare_court_data.create_df_from_tar('data/opinions_wash.tar.gz')
opinion_rdd = sc.parallelize(opinion_lst)

# html parsing
df = ps.DataFrame(opinion_lst)
doc = df['html_columbia'][0]
soup = BeautifulSoup(doc, 'lxml')
print(soup.get_text())
