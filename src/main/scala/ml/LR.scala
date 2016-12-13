package main.scala.ml

import org.apache.spark.sql.Dataset
import main.scala.schema.Observation
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.tuning.ParamGridBuilder

object LR extends Classifier {
    val algorithmName = "Logistic Regression Classifier"

    def generate(data: Dataset[Observation], target: String):
    Transformer = {
        val lg = new LogisticRegression()
        val paramGrid = getTuningParams(lg)
        return generateModel(data, target, lg, algorithmName, Some(paramGrid))
    }

    def getTuningParams(lg: LogisticRegression): ParamGridBuilder = {
        val paramGrid = new ParamGridBuilder()
            .addGrid(lg.elasticNetParam, Array(.4, .8))
            .addGrid(lg.maxIter, Array(10, 30, 60))
        return paramGrid
    }
}
