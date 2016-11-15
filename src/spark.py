from src import prepare_court_data
import pyspark as ps
from bs4 import BeautifulSoup
import pandas as pd


opinion_lst = prepare_court_data.create_df_from_tar('data/opinions_wash.tar.gz')
opinion_rdd = sc.parallelize(opinion_lst, 15)

df_columns = ['cluster', 'parsed_columbia', 'parsed_lawbox', 'per_curiam', 'opinions_cited', 'type']
fields = [ps.sql.types.StructField(field_name, ps.sql.types.StringType(), True) for field_name in df_columns]
schema = ps.sql.types.StructType(fields)

reduced_rdd = opinion_rdd.map(lambda row: [row.get(key, '') for key in df_columns])
opinion_df = spark.createDataFrame(reduced_rdd, schema)

test_parse = sc.parallelize(opinion_rdd.take(10), 15)
test_rdd = test_parse.map(lambda row: [row.get(key,'') for key in df_columns])
test_df = spark.createDataFrame(test_rdd, schema)

