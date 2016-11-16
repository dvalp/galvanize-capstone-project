from src import prepare_court_data
import pyspark as ps
from bs4 import BeautifulSoup
import pandas as pd
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, NGram, CountVectorizer, IDF


# create an RDD from the data
opinion_lst = prepare_court_data.create_df_from_tar('data/opinions_wash.tar.gz')
opinion_rdd = sc.parallelize(opinion_lst, 15)

# define the subset of columns to use and write it into a schema
df_columns = ['cluster_id', 'resource_id', 'parsed_text', 'per_curiam', 'opinions_cited', 'type']
fields = [ps.sql.types.StructField(field_name, ps.sql.types.StringType(), True) for field_name in df_columns]
schema = ps.sql.types.StructType(fields)

# create the dataframe from just the columns we want
reduced_rdd = opinion_rdd.map(lambda row: [row.get(key, '') for key in df_columns])
opinion_df = spark.createDataFrame(reduced_rdd, schema)

# create a test dataframe
test_parse = sc.parallelize(opinion_rdd.take(10), 15)
test_rdd = test_parse.map(lambda row: [row.get(key,'') for key in df_columns])
test_df = spark.createDataFrame(test_rdd, schema)

# define the transformations
# tokenizer = Tokenizer(inputCol='parsed_text', outputCol='tokens')
regexTokenizer = RegexTokenizer(inputCol="parsed_text", outputCol="tokens", pattern="\\W", minTokenLength=3)
remover = StopWordsRemover(inputCol='tokens', outputCol='tokens_stop')
bigram = NGram(inputCol='tokens', outputCol='bigrams', n=2)
trigram = NGram(inputCol='tokens', outputCol='trigrams', n=3)

# run transformations
tokens_df = regexTokenizer.transform(test_df)
tokens_df = remover.transform(tokens_df)
tokens_df = bigram.transform(tokens_df)
tokens_df = trigram.transform(tokens_df)

op_tokens_df = regexTokenizer.transform(opinion_df)
op_tokens_df = remover.transform(op_tokens_df)
op_tokens_df = bigram.transform(op_tokens_df)
op_tokens_df = trigram.transform(op_tokens_df)

# CountVectorizer
cv = CountVectorizer(inputCol='tokens_stop', outputCol='token_countvector', minDF=2.0)
opinion_cv_model = cv.fit(op_tokens_df)
op_tokens_df = opinion_cv_model.transform(op_tokens_df)

# IDF
idf = IDF(inputCol='token_countvector', outputCol='token_idf', minDocFreq=2)
opinion_idf_model = idf.fit(op_tokens_df)
