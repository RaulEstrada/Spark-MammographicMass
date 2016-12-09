package main.scala.ml

import org.apache.spark.sql.Dataset
import main.scala.schema.Observation
import org.apache.spark.sql.SparkSession

object Ensembler {

    val accumulatorName = "Wrong Prediction Accumulator"

    def ensemble(data: Dataset[Observation], target: String, session: SparkSession) {
        // Generate and train individual classifiers and print their individual test errors
        val rf = RF.generate(data, target)
        val lr = LR.generate(data, target)
        val gbt = GBT.generate(data, target)
        val mpc = MPC.generate(data, target)

        val context = session.sparkContext
        context.broadcast(rf)
        context.broadcast(lr)
        context.broadcast(gbt)
        context.broadcast(mpc)

        // Obtain error of classifier
        val Array(trainingData, testData) = data.randomSplit(Array(.7, .3))
        val wrong = context.longAccumulator(accumulatorName)
    }
}
