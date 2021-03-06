package main.scala.ml

import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.classification.GBTClassifier
import main.scala.schema.Observation
import org.apache.spark.ml.tuning.ParamGridBuilder

object GBT extends Classifier {
    val algorithmName = "Gradient-Boosted Tree"

    def generate(data: Dataset[Observation], target: String): CrossValidatorModel = {
        val gbt = new GBTClassifier()
        val paramGrid = getTuningParams(gbt)
        val model = generateModel(data, target, gbt, algorithmName, Some(paramGrid))
        return model
    }

    def getTuningParams(gbt: GBTClassifier): ParamGridBuilder = {
        val paramGrid = new ParamGridBuilder()
            .addGrid(gbt.maxIter, Array(5, 10, 20))
            .addGrid(gbt.maxDepth, Array(5, 10, 15))
        return paramGrid
    }
}
