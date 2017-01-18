from src.prepare_court_data import import_dataframe, reverse_stem
from src.ml_transformer import Stemming_Transformer
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pyspark as ps
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, NGram, \
        CountVectorizer, IDF, Word2Vec
from pyspark.sql.functions import udf, col, explode, collect_list, to_date, concat
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, \
        FloatType, ArrayType, BooleanType
from nltk.stem import SnowballStemmer


# Import json objects from tar file
opinion_df = import_dataframe(spark, 'opinion')
docket_df = import_dataframe(spark, 'docket')
cluster_df = import_dataframe(spark, 'cluster')

# Setup pipeline for adding ML features - tokens, stems, n-grams, tf, tfidf, word2vec
# tokenizer = Tokenizer(inputCol='parsed_text', outputCol='tokens')
tokenizer = RegexTokenizer(inputCol="parsed_text", outputCol="raw_tokens", pattern="\\W", minTokenLength=3)
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='tokens_stop')
stemmer = Stemming_Transformer(inputCol=remover.getOutputCol(), outputCol='tokens')
bigram = NGram(inputCol=stemmer.getOutputCol(), outputCol='bigrams', n=2)
trigram = NGram(inputCol=stemmer.getOutputCol(), outputCol='trigrams', n=3)
cv = CountVectorizer(inputCol=stemmer.getOutputCol(), outputCol='token_countvector', minDF=10.0)
idf = IDF(inputCol=cv.getOutputCol(), outputCol='token_idf', minDocFreq=10)
w2v_2d = Word2Vec(vectorSize=2, minCount=2, inputCol=stemmer.getOutputCol(), outputCol='word2vec_2d')
w2v_large = Word2Vec(vectorSize=250, minCount=2, inputCol=stemmer.getOutputCol(), outputCol='word2vec_large')

pipe = Pipeline(stages=[tokenizer, remover, stemmer, cv, idf, w2v_2d, w2v_large])

# Use the pipeline to fit a model
model = pipe.fit(opinion_df)

# Use the model to transform the data
df_transformed = model.transform(opinion_df)

# retrieve top 10 number of words for the document, assumes existence of 'row' containg one row from the dataframe
np.array(opinion_cv_model.vocabulary)[row['token_idf'].indices[np.argsort(row['token_idf'].values)]][:-11:-1]

# save and retrieve dataframe
opiniondf_w2vlarge.write.save('data/opinions-spark-data.json', format='json', mode='overwrite')
opinion_loaded = spark.read.json('data/opinions-spark-data.json')

# extract the vector from a specific document and take the squared distance or cosine similarity for all other documents, show the ten nearest
ref_vec = df_transformed.filter(df_transformed.resource_id == '1390131').first()['word2vec_large']

udf_squared_distance = udf(lambda cell: float(ref_vec.squared_distance(cell)), FloatType())
df_transformed \
        .withColumn('squared_distance', udf_squared_distance(df_transformed.word2vec_large)) \
        .sort(col('squared_distance'), ascending=True) \
        .select('resource_id', 'squared_distance')
        .show(10)

udf_cos_sim = udf(lambda cell: float(ref_vec.dot(cell) / (ref_vec.norm(2) * cell.norm(2))), FloatType())
df_transformed \
        .withColumn('cos_similarity', udf_cos_sim(df_transformed.word2vec_large)) \
        .sort(col('cos_similarity'), ascending=False) \
        .select('resource_id', 'cos_similarity')
        .show(10)

# create a list of terms connected to their stems
df_wordcount = spark \
        .createDataFrame(df_transformed.select(explode(df_transformed.tokens_stop).alias('term')) \
        .groupBy('term') \
        .agg({"*": "count"}) \
        .collect())
df_stems = df_wordcount \
        .withColumn('stem', udf(lambda term: opinion_stemm.stem(term), StringType())(col('term'))) \
        .groupBy('stem') \
        .agg(collect_list('term').alias('terms'))
df_stems \
        .select('terms') \
        .filter(df_stems.stem == opinion_stemm.stem('artful')) \
        .first()[0]

# create a count for each opinion of the number of times it has been cited by other Washington opinions
df_citecount = spark.createDataFrame(
        df_transformed.select(explode(df_transformed.opinions_cited).alias('cites')) \
                .groupBy('cites') \
                .count() \
                .collect())
df_citecount.orderBy('count', ascending=False).show()

