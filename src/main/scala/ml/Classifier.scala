package main.scala.ml

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import main.scala.schema.Observation

class Classifier {
    val labelCol = "label"
    val featuresCol = "features"
    val predictionCol = "prediction"
    val metric = "accuracy"
    val percentageTrainingData = .7

    def indexer(data: Dataset[Observation], target: String): StringIndexerModel = {
        return new StringIndexer()
            .setInputCol(target)
            .setOutputCol(labelCol)
            .fit(data)
    }

    def features(data: Dataset[Observation], target: String): VectorAssembler = {
        return new VectorAssembler()
            .setInputCols(Observation.getFeaturesArray(target))
            .setOutputCol(featuresCol)
    }

    def splits(data: Dataset[Observation]): Array[Dataset[Observation]] = {
        return data.randomSplit(Array(percentageTrainingData, (1 - percentageTrainingData)))
    }

    def evaluateModel(algorithm: String, result: DataFrame) {
        val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol(labelCol)
            .setPredictionCol(predictionCol)
            .setMetricName(metric)
        val accuracy = evaluator.evaluate(result)
        println(algorithm + " test error: " + (1 - accuracy))
    }

    def generateModel(data: Dataset[Observation], target: String,
    stage: PipelineStage, algorithm: String): PipelineModel = {
        val Array(trainingData, testData) = splits(data)
        val indexerStage = indexer(data, target)
        val featuresStage = features(data, target)
        val pipeline = new Pipeline()
            .setStages(Array(indexerStage, featuresStage, stage))
        val model = pipeline.fit(trainingData)
        val result = model.transform(testData)
        evaluateModel(algorithm, result)
        return model
    }

}
