package main.scala.ml

import org.apache.spark.sql.Dataset
import main.scala.schema.Observation
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel

object LR {

    def generate(data: Dataset[Observation], target: String):
    PipelineModel = {
        // Convert String values that are part of a look-up into categorical
        // indices with a few discrete values. Index the severity category
        val indexer = new StringIndexer()
            .setInputCol(target)
            .setOutputCol("label")
            .fit(data)

        // Create a column that is a vector of all the features/predictor values
        val features = new VectorAssembler()
            .setInputCols(Observation.getFeaturesArray(target))
            .setOutputCol("features")

        println("LR " + target)

        val logisticRegr = new LogisticRegression()
            .setElasticNetParam(0.8)

        // Assemble the machine learning pipeline
        val pipeline = new Pipeline()
            .setStages(Array(indexer, features, logisticRegr))

        val model = pipeline.fit(data)
        val lrModel = model.stages(2).asInstanceOf[LogisticRegressionModel]
        println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
        return model
    }
}
