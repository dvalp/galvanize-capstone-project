#!/bin/bash
export PYSPARK_PYTHON=/home/ratboy/.virtualenvs/capstone/bin/python3
export PYSPARK_DRIVER_PYTHON=ipython 

/home/ratboy/build/spark/bin/pyspark \
	--master local[6] \
	--executor-memory 20G \
	--driver-memory 20G \
	--packages com.databricks:spark-csv_2.11:1.5.0 \
	--packages com.amazonaws:aws-java-sdk-pom:1.10.34 \
	--packages org.apache.hadoop:hadoop-aws:2.7.3
