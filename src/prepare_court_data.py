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

def import_dataframe(spark, doc_type):
    '''
    Import the data from a tar.gz file. Use a generator so the data isn't loaded until necessary.

    The tarfile module provides a TarInfo object that can be used to identify each file for extraction.
    The extractfile() method returns a file object that has a read() method which returns a byte 
    representation of the file which can be decoded into a string. The string can then be translated
    into a dict by json.loads(). Each json object in dict form can be read into the dataframe.

    Pass in a Spark session object so the data can be loaded into the current session.

    The schema is important because inferring the data type from a dict is deprecated in Spark and 
    sometimes assumes the wrong type.
    '''
    doc_path, schema = get_doc_schema(doc_type)

    with tarfile.open(doc_path, mode='r:gz') as tf:
        json_gen = (json.loads(tf.extractfile(f).read().decode()) for f in tf)
        raw_dataframe = spark.createDataFrame(json_gen, schema)

    return raw_dataframe

def get_doc_schema(doc_type):
    '''
    Return the document schema and path for importing data files into Spark. Removed from import 
    to reduce redundancy and limit places code needs to change for imports. 

    Will return None for path and schema if unknown type is passed.
    '''
    if doc_type == 'opinion':
        path = 'data/opinions_wash.tar.gz'
        schema = StructType([
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
            StructField('type', StringType(), True)])
        
    elif doc_type == 'docket':
        path = 'data/dockets_wash.tar.gz'
        schema = StructType([
            StructField('date_reargued', StringType(), True),
            StructField('date_filed', StringType(), True),
            StructField('date_argued', StringType(), True),
            StructField('date_terminated', StringType(), True),
            StructField('date_cert_denied', StringType(), True),
            StructField('absolute_url', StringType(), True),
            StructField('source', StringType(), True),
            StructField('jury_demand', StringType(), True),
            StructField('cause', StringType(), True),
            StructField('date_cert_granted', StringType(), True),
            StructField('case_name_full', StringType(), True),
            StructField('date_modified', StringType(), True),
            StructField('clusters', ArrayType(StringType()), True),
            StructField('date_last_filing', StringType(), True),
            StructField('court', StringType(), True),
            StructField('nature_of_suit', StringType(), True),
            StructField('date_reargument_denied', StringType(), True),
            StructField('filepath_ia', StringType(), True),
            StructField('date_created', StringType(), True),
            StructField('referred_to', StringType(), True),
            StructField('resource_uri', StringType(), True),
            StructField('docket_number', StringType(), True),
            StructField('filepath_local', StringType(), True),
            StructField('case_name_short', StringType(), True),
            StructField('jurisdiction_type', StringType(), True),
            StructField('blocked', BooleanType(), True),
            StructField('assigned_to', StringType(), True),
            StructField('pacer_case_id', StringType(), True),
            StructField('date_blocked', StringType(), True),
            StructField('audio_files', StringType(), True),
            StructField('slug', StringType(), True),
            StructField('case_name', StringType(), True)])

    elif doc_type == 'cluster':
        path = 'data/clusters_wash.tar.gz'
        schema = StructType([
            StructField('neutral_cite', StringType(), True),
            StructField('panel', StringType(), True),
            StructField('specialty_cite_one', StringType(), True),
            StructField('westlaw_cite', StringType(), True),
            StructField('lexis_cite', StringType(), True),
            StructField('absolute_url', StringType(), True),
            StructField('source', StringType(), True),
            StructField('federal_cite_two', StringType(), True),
            StructField('judges', StringType(), True),
            StructField('date_filed', StringType(), True),
            StructField('case_name_full', StringType(), True),
            StructField('blocked', StringType(), True),
            StructField('date_modified', StringType(), True),
            StructField('scdb_votes_minority', StringType(), True),
            StructField('case_name', StringType(), True),
            StructField('syllabus', StringType(), True),
            StructField('state_cite_two', StringType(), True),
            StructField('federal_cite_three', StringType(), True),
            StructField('nature_of_suit', StringType(), True),
            StructField('date_created', StringType(), True),
            StructField('state_cite_three', StringType(), True),
            StructField('scdb_id', StringType(), True),
            StructField('scdb_votes_majority', StringType(), True),
            StructField('resource_uri', StringType(), True),
            StructField('procedural_history', StringType(), True),
            StructField('posture', StringType(), True),
            StructField('non_participating_judges', StringType(), True),
            StructField('state_cite_regional', StringType(), True),
            StructField('state_cite_one', StringType(), True),
            StructField('attorneys', StringType(), True),
            StructField('case_name_short', StringType(), True),
            StructField('docket', StringType(), True),
            StructField('sub_opinions', ArrayType(StringType()), True),
            StructField('scdb_decision_direction', StringType(), True),
            StructField('citation_count', IntegerType(), True),
            StructField('date_blocked', StringType(), True),
            StructField('precedential_status', StringType(), True),
            StructField('slug', StringType(), True),
            StructField('federal_cite_one', StringType(), True),
            StructField('citation_id', StringType(), True),
            StructField('scotus_early_cite', StringType(), True)])
    else:
        path = None
        schema = None

    return path, schema
