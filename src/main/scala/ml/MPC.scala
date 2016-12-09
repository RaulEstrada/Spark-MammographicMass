package main.scala.ml

import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.IndexToString
import main.scala.schema.Observation

object MPC {
    def generate(data: Dataset[Observation], target: String): PipelineModel = {
        val indexer = new StringIndexer()
            .setInputCol(target)
            .setOutputCol("label")
            .fit(data)

        val features = new VectorAssembler()
            .setInputCols(Observation.getFeaturesArray(target))
            .setOutputCol("features")

        val Array(trainingData, testData) = data.randomSplit(Array(.7, .3))
        println("MPC " + target + " - Training data: " +
            trainingData.count() + " - Test data: " + testData.count())

        val layers = Array[Int](4, 15, 2)
        val mpc = new MultilayerPerceptronClassifier()
            .setLayers(layers)

        val pipeline = new Pipeline()
            .setStages(Array(indexer, features, mpc))
        val model = pipeline.fit(trainingData)

        val result = model.transform(testData)
        val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(result)
        println("Test error of Multilayer Perceptron Classifier: " + (1 - accuracy))
        return model
    }
}
