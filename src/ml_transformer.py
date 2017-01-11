from pyspark.ml.util import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from nltk.stem import SnowballStemmer

# Custom stemming transformer class for pyspark
class stemming_transformer(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(stemming_transformer, self).__init__()

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        pass

    def _transform(self, dataset):
        pass
