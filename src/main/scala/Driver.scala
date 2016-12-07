package main.scala

import org.apache.spark.sql.SparkSession
import main.scala.preprocessing.DataPreProcessor
import main.scala.ml.RF
import main.scala.ml.LR
import main.scala.ml.GBT

object Driver {
    def main(args: Array[String]) {
        if (args.length < 1) {
            println("MammographicMass Driver usage: <file>")
            System.exit(-1)
        }
        // Entry point of Spark SQL
        val session = SparkSession.builder().getOrCreate();
        var observations = DataPreProcessor.preprocess(session, args(0))
        RF.generate(observations, "Severity")

        val Array(trainingData, testData) = observations.randomSplit(Array(.7, .3))
        val mlr = LR.generate(trainingData, "Severity")
        val countWrong = mlr.transform(testData).filter("Severity <> prediction").count()
        println("Test error: " + countWrong.toDouble/testData.count().toDouble)

        GBT.generate(observations, "Severity")
    }
}
