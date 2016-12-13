package main.scala.ml

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.Dataset
import main.scala.schema.Observation
import org.apache.spark.ml.tuning.ParamGridBuilder

object RF extends Classifier {
    var numberOfTrees = 60
    val impurity = "gini"
    val maxDepth = 30
    val featureSubsetStrategy = "auto"
    val algorithmName = "Random Forest"

    def generate(data: Dataset[Observation], target: String): Transformer = {
        val rf = new RandomForestClassifier()
            .setImpurity(impurity)
            .setFeatureSubsetStrategy(featureSubsetStrategy)
        val paramBuilder = getTuningParams(rf)
        return generateModel(data, target, rf, algorithmName, Some(paramBuilder))
    }

    def getTuningParams(rf: RandomForestClassifier): ParamGridBuilder = {
        val paramGrid = new ParamGridBuilder()
            .addGrid(rf.numTrees, Array(10, 30, 60, 80))
            .addGrid(rf.maxDepth, Array(5, 10, 15, 30))
        return paramGrid
    }
}
