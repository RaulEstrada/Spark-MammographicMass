package main.scala.ml

import org.apache.spark.sql.Dataset
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.Transformer
import main.scala.schema.Observation

object MPC extends Classifier {
    val algorithmName = "Multi-Layer Perceptron Classifier"

    def generate(data: Dataset[Observation], target: String): Transformer = {
        val layers = Array[Int](4, 15, 2)
        val mpc = new MultilayerPerceptronClassifier()
            .setLayers(layers)
        return generateModel(data, target, mpc, algorithmName, None)
    }
}
