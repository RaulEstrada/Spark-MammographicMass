package main.scala.ml

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.sql.Dataset
import main.scala.schema.Observation

object RF {
    var numberOfTrees = 60
    val impurity = "gini"
    val maxDepth = 30
    val featureSubsetStrategy = "auto"
    val seed = 1234

    def generate(data: Dataset[Observation], target: String): PipelineModel = {
        val dataNotNA = data.filter(data.col(target).isNotNull)

        // Convert String values that are part of a look-up into categorical
        // indices with a few discrete values. Index the severity category
        val indexer = new StringIndexer()
            .setInputCol(target)
            .setOutputCol("Indexed")
            .fit(dataNotNA)

        // Create a column that is a vector of all the features/predictor values
        val features = new VectorAssembler()
            .setInputCols(Observation.getFeaturesArray(target))
            .setOutputCol("Features")

        // Split data into training and test subsets (70% training, 30% test)
        val Array(trainingData, testData) = dataNotNA.randomSplit(Array(.7, .3))
        println("RF " + target + " - " + numberOfTrees + " trees - Training data: " +
            trainingData.count() + " - Test data: " + testData.count())

        // Train a RandomForest
        val randomForest = new RandomForestClassifier()
            .setLabelCol("Indexed")
            .setFeaturesCol("Features")
            .setNumTrees(numberOfTrees)
            .setImpurity(impurity)
            .setMaxDepth(maxDepth)
            .setFeatureSubsetStrategy(featureSubsetStrategy)
            .setSeed(seed)

        // Convert indexed predicted labels back to original labels
        val labelConverter = new IndexToString()
            .setInputCol("prediction")
            .setOutputCol("Predicted")
            .setLabels(indexer.labels)

        // Assemble the machine learning pipeline
        val pipeline = new Pipeline()
            .setStages(Array(indexer, features, randomForest, labelConverter))

        // Train the model
        val model = pipeline.fit(trainingData)

        // Predict values
        val predictions = model.transform(testData)

        //Evaluation
        val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("Indexed")
            .setPredictionCol("prediction")
            .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println("Random Forest with test error: " + (1.0 - accuracy))

        return model
    }
}
