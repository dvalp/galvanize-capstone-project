from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol

# Custom stemming transformer class for pyspark
class stemming_transformer(Transformer, HasInputCol, HasOutputCol):
    pass
