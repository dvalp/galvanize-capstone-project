from src import prepare_court_data
import pyspark as ps
from bs4 import BeautifulSoup
import pandas as pd


opinion_lst = prepare_court_data.create_df_from_tar('data/opinions_wash.tar.gz')
opinion_rdd = sc.parallelize(opinion_lst, 15)

test_parse = sc.parallelize(opinion_rdd.take(10), 15)

