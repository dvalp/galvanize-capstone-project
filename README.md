# NLP Analysis of Washington State Supreme Court Opinions

Galvanize Data Science Immersive Capstone Project using Apache Spark and NLP to analyze Washington State Supreme Court opinions. The data has been made available by the [Free Law Project](https://free.law/) via the [CourtListener](https://www.courtlistener.com/) website. CourtListener provides the data via an API or as a bulk download of all the documents in a compressed archive of individual files, each containing a single JSON object for one court opinion.

[Apache Spark](https://spark.apache.org/) an engine for large scale data processing that makes it possible to work on large amounts of data in parallel wherever possible. Where Spark does not have native functions to implement machine learning algorithms, it can also run external functions in parallel (in this case functions are imported from Python.

Where functions are not available in Spark, I have primarily made use of the [Natural Language Toolkit (NLTK)](http://www.nltk.org/) and [Scikit-Learn](http://scikit-learn.org/) to process my data.
