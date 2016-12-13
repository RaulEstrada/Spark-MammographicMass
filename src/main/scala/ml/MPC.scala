package main.scala.ml

import org.apache.spark.sql.Dataset
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.Transformer
import main.scala.schema.Observation
import org.apache.spark.ml.tuning.ParamGridBuilder

object MPC extends Classifier {
    val algorithmName = "Multi-Layer Perceptron Classifier"

    def generate(data: Dataset[Observation], target: String): Transformer = {
        val mpc = new MultilayerPerceptronClassifier()
        val paramGrid = getTuningParams(mpc)
        return generateModel(data, target, mpc, algorithmName, Some(paramGrid))
    }

    def getTuningParams(mpc: MultilayerPerceptronClassifier): ParamGridBuilder = {
        val paramGrid = new ParamGridBuilder()
            .addGrid(mpc.maxIter, Array(10, 30, 60))
            .addGrid(mpc.layers, Array(
                Array[Int](4, 15, 2),
                Array[Int](4, 10, 2),
                Array[Int](4, 3, 2),
                Array[Int](4, 7, 2)))
        return paramGrid
    }
}
