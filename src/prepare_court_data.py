import pandas as pd
import numpy as np
import tarfile
import json
from bs4 import BeautifulSoup
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, \
        FloatType, ArrayType, BooleanType


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

def import_opinions_as_dataframe(tf_path = 'data/opinions_wash.tar.gz'):
    '''
    Import the data from a tar.gz file. Use a generator so the data isn't loaded until necessary.

    The tarfile module provides a TarInfo object that can be used to identify each file for extraction.
    The extractfile() method returns a file object that has a read() method which returns a byte 
    representation of the file which can be decoded into a string. The string can then be translated
    into a dict by json.loads(). Each json object in dict form can be read into the dataframe.

    The schema is important because inferring the data type from a dict is deprecated in Spark and 
    sometimes assumes the wrong type.
    '''

    fields = [
            StructField('absolute_url', StringType(), True),
            StructField('author', StringType(), True),
            StructField('cluster', StringType(), True),
            StructField('date_created', StringType(), True),
            StructField('date_modified', StringType(), True),
            StructField('download_url', StringType(), True),
            StructField('extracted_by_ocr', BooleanType(), True),
            StructField('html', StringType(), True),
            StructField('html_columbia', StringType(), True),
            StructField('html_lawbox', StringType(), True),
            StructField('html_with_citations', StringType(), True),
            StructField('joined_by', ArrayType(StringType()), True),
            StructField('local_path', StringType(), True),
            StructField('opinions_cited', ArrayType(StringType()), True),
            StructField('per_curiam', BooleanType(), True), 
            StructField('plain_text', StringType(), True),
            StructField('resource_uri', StringType(), True),
            StructField('sha1', StringType(), True),
            StructField('type', StringType(), True)]
    schema = StructType(fields)

    with tarfile.open(tf_path, mode='r:gz') as tf:
        # create a generator expression to avoid loading everything into memory at once
        opinions = (json.loads(tf.extractfile(f).read().decode()) for f in tf)
        raw_dataframe = spark.createDataFrame(opinions, schema)

    return raw_dataframe
