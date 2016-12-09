package main.scala.ml

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.Dataset
import main.scala.schema.Observation

object RF extends Classifier {
    var numberOfTrees = 60
    val impurity = "gini"
    val maxDepth = 30
    val featureSubsetStrategy = "auto"
    val algorithmName = "Random Forest"

    def generate(data: Dataset[Observation], target: String): PipelineModel = {
        val rf = new RandomForestClassifier()
            .setNumTrees(numberOfTrees)
            .setImpurity(impurity)
            .setMaxDepth(maxDepth)
            .setFeatureSubsetStrategy(featureSubsetStrategy)
        return generateModel(data, target, rf, algorithmName)
    }
}
