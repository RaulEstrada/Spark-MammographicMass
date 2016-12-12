package main.scala.ml

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import main.scala.schema.Observation
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.DenseVector

object Ensembler {

    val accumulatorName = "Wrong Prediction Accumulator"

    def ensemble(data: Dataset[Observation], target: String, session: SparkSession) {
        import session.implicits._
        // Generate and train individual classifiers and print their individual test errors
        val rf = RF.generate(data, target)
        val lr = LR.generate(data, target)
        val gbt = GBT.generate(data, target)
        val mpc = MPC.generate(data, target)

        val context = session.sparkContext
        val ensamble = context.parallelize(Array(rf, lr, gbt, mpc)).collect()

        // Obtain error of classifier
        val Array(trainingData, testData) = data.randomSplit(Array(.7, .3))
        context.broadcast(testData)
        val wrong = context.longAccumulator(accumulatorName)
        var collectedPredictions = ensamble.map(x => x.transform(testData))
            .reduce{ (a: DataFrame, b: DataFrame) =>
                a.select("label", "features", "prediction").union(b.select("label", "features", "prediction"))
            }.map { row =>
                ((row.getDouble(0), row.getAs[DenseVector](1)), Array[Double](row.getDouble(2)))
            }.rdd.groupByKey().map{x =>
                var modeValue = mode(x._2, session)
                var label = x._1._1
                println("LABEL: " + label + "\tMODE: " + modeValue)
                if (modeValue != label) {
                    1
                } else {
                    0
                }
            }.reduce(_+_)
        println("ENSAMBLE approach Test Error: " + collectedPredictions.toDouble/testData.count())
    }

    def mode(data: Iterable[Array[Double]], session: SparkSession): Double = {
        var collected = data.map(x => x).reduce((a, b) => a ++ b)
        println("COLLECTED: " + collected.mkString(" "))
        var occurrences = Map[Double, Int]()
        var element = 0.0
        for (element <- collected) {
            var newValue = occurrences.getOrElse(element, 0) + 1
            occurrences = occurrences + (element -> newValue)
        }
        var mode = -1.0
        var cnt = 0
        var key = 0.0
        for (key <- occurrences.keys) {
            if (occurrences(key) > cnt) {
                mode = key
                cnt = occurrences(key)
            }
        }
        return mode
    }
}
