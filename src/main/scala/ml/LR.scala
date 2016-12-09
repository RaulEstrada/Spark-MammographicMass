package main.scala.ml

import org.apache.spark.sql.Dataset
import main.scala.schema.Observation
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.PipelineModel

object LR extends Classifier {
    val algorithmName = "Logistic Regression Classifier"

    def generate(data: Dataset[Observation], target: String):
    PipelineModel = {
        val lg = new LogisticRegression()
            .setElasticNetParam(0.8)
        return generateModel(data, target, lg, algorithmName)
    }
}
