package main.scala.ml

import org.apache.spark.ml.Transformer
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.classification.GBTClassifier
import main.scala.schema.Observation

object GBT extends Classifier {
    val algorithmName = "Gradient-Boosted Tree"

    def generate(data: Dataset[Observation], target: String): Transformer = {
        val gbt = new GBTClassifier()
        return generateModel(data, target, gbt, algorithmName, None)
    }
}
