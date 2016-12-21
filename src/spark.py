from src import prepare_court_data
import pyspark as ps
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, NGram, \
        CountVectorizer, IDF, Word2Vec
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, \
        FloatType, ArrayType, BooleanType
from nltk.stem import SnowballStemmer
from pyspark.sql.functions import udf, col, explode, collect_list


# create an RDD from the data, choose number of rows to include
opinion_lst = prepare_court_data.create_df_from_tar('data/opinions_wash.tar.gz', 100)
opinion_rdd = sc.parallelize(opinion_lst, 15)

# define the subset of columns to use and write it into a schema
df_columns = ['cluster_id', 'resource_id', 'parsed_text', 'per_curiam', 'opinions_cited', 'type']
fields = [StructField('cluster_id', IntegerType(), True), 
        StructField('resource_id', IntegerType(), True),
        StructField('parsed_text', StringType(), True),
        StructField('per_curiam', BooleanType(), True),
        StructField('opinions_cited', ArrayType(IntegerType()), True),
        StructField('type', StringType(), True)]
schema = StructType(fields)

# create the dataframe from just the columns we want
reduced_rdd = opinion_rdd.map(lambda row: [row.get(key, '') for key in df_columns])
opinion_df = spark.createDataFrame(reduced_rdd, schema)

# Parse tokens from text, remove stopwords
# tokenizer = Tokenizer(inputCol='parsed_text', outputCol='tokens')
regexTokenizer = RegexTokenizer(inputCol="parsed_text", outputCol="raw_tokens", pattern="\\W", minTokenLength=3)
remover = StopWordsRemover(inputCol='raw_tokens', outputCol='tokens_stop')
opiniondf_tokenized = regexTokenizer.transform(opinion_df)
opiniondf_nostop = remover.transform(opiniondf_tokenized)

# use a udf to stem the tokens using the nltk SnowballStemmer with the English dictionary
opinion_stemm = SnowballStemmer('english')
udfStemmer = udf(lambda tokens: [opinion_stemm.stem(word) for word in tokens], ArrayType(StringType()))
opiniondf_stemmed = opiniondf_nostop.withColumn('tokens', udfStemmer(opiniondf_nostop.tokens_stop))

# create n-grams
bigram = NGram(inputCol='tokens', outputCol='bigrams', n=2)
trigram = NGram(inputCol='tokens', outputCol='trigrams', n=3)
opiniondf_bigram = bigram.transform(opiniondf_stemmed)
opiniondf_trigram = trigram.transform(opiniondf_bigram)

# CountVectorizer
cv = CountVectorizer(inputCol='tokens', outputCol='token_countvector', minDF=10.0)
opinion_cv_model = cv.fit(opiniondf_trigram)
opiniondf_countvector = opinion_cv_model.transform(opiniondf_trigram)

# IDF
idf = IDF(inputCol='token_countvector', outputCol='token_idf', minDocFreq=10)
opinion_idf_model = idf.fit(opiniondf_countvector)
opiniondf_idf = opinion_idf_model.transform(opiniondf_countvector)

# Word2Vec
w2v_2d = Word2Vec(vectorSize=2, minCount=2, inputCol='tokens', outputCol='word2vec_2d')
w2v_large = Word2Vec(vectorSize=250, minCount=2, inputCol='tokens', outputCol='word2vec_large')
w2v2d_model = w2v_2d.fit(opiniondf_idf)
w2vlarge_model = w2v_large.fit(opiniondf_idf)
opiniondf_w2v2d = w2v2d_model.transform(opiniondf_idf)
opiniondf_w2vlarge = w2vlarge_model.transform(opiniondf_w2v2d)

# retrieve top 10 number of words for the document, assumes existence of 'row' containg one row from the dataframe
np.array(opinion_cv_model.vocabulary)[row['token_idf'].indices[np.argsort(row['token_idf'].values)]][:-11:-1]

# save and retrieve dataframe
opiniondf_w2vlarge.write.save('data/opinions-spark-data.json', format='json', mode='overwrite')
opinion_loaded = spark.read.json('data/opinions-spark-data.json')

# extract the vector from a specific document and take the cosine similarity for all other documents, show the ten nearest
ref_vec = opiniondf_w2vlarge.filter(opiniondf_w2vlarge.resource_id == '3990749').first()['word2vec_large']

udfSqDist = udf(lambda cell: float(ref_vec.squared_distance(cell)), FloatType())
opiniondf_w2vlarge.withColumn('squared_distance', udfSqDist(opiniondf_w2vlarge.word2vec_large)).sort(col('squared_distance'), ascending=True).select('cluster_id', 'resource_id', 'squared_distance').show(10)

udf_cos_sim = udf(lambda cell: float(ref_vec.dot(cell) / (ref_vec.norm() * cell.norm())), FloatType())
opinion_df.withColumn('cos_similarity', udf_cos_sim(opinion_df.word2vec_large)).sort(col('cos_similarity'), ascending=False).select('cluster_id', 'resource_id', 'cos_similarity').show(10)

# create a list of terms connected to their stems
df_wordcount = spark.createDataFrame(opinion_df.select(explode(opinion_df.tokens_stop).alias('term')).groupBy('term').agg({"*": "count"}).collect())
df_stems = df_wordcount.withColumn('stem', udf(lambda term: opinion_stemm.stem(term), StringType())(col('term'))).groupBy('stem').agg(collect_list('term').alias('terms'))
df_stems.select('terms').filter(df_stems.stem == opinion_stemm.stem('artful')).first()[0]

# create a count for each opinion of the number of times it has been cited by other Washington opinions
df_citecount = spark.createDataFrame(opinion_df.select(explode(opinion_df.opinions_cited).alias('cites')).groupBy('cites').agg({"*": "count"}).collect())
df_citecount.orderBy('count(1)', ascending=False).show()
