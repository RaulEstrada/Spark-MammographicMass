package main.scala.ml

import org.apache.spark.sql.Dataset
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.Transformer
import main.scala.schema.Observation

object NB extends Classifier {
    val algorithmName = "Naive Bayes"

    def generate(data: Dataset[Observation], target: String): Transformer = {
        val nb = new NaiveBayes()
        return generateModel(data, target, nb, algorithmName, None)
    }
}
