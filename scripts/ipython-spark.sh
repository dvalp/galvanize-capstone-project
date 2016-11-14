#!/bin/bash
export PYSPARK_PYTHON=/usr/bin/python3 
export PYSPARK_DRIVER_PYTHON=ipython 

/usr/local/spark/bin/pyspark \
	--master local[5] \
	--executor-memory 12G \
	--driver-memory 12G \
	--packages com.databricks:spark-csv_2.11:1.5.0 \
        --packages com.amazonaws:aws-java-sdk-pom:1.10.34 \
        --packages org.apache.hadoop:hadoop-aws:2.7.3
