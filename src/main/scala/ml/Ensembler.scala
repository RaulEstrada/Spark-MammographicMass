package main.scala.ml

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import main.scala.schema.Observation
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD

object Ensembler {

    val accumulatorName = "Wrong Prediction Accumulator"

    def ensemble(data: Dataset[Observation], target: String, session: SparkSession) {
        import session.implicits._

        val ensamble = generateEnsamble(data, target, session)
        val Array(trainingData, testData) = data.randomSplit(Array(.7, .3), seed = Classifier.SEED)
        val collectedPredictions = predict(testData, ensamble, session)
        computeEnsembleError(collectedPredictions, testData)
    }

    def generateEnsamble(data: Dataset[Observation], target: String,
    session: SparkSession): Array[Transformer] = {
        // Generate and train individual classifiers and print their individual test errors
        val rf = RF.generate(data, target)
        val lr = LR.generate(data, target)
        val gbt = GBT.generate(data, target)
        val mpc = MPC.generate(data, target)
        //val nb = NB.generate(data, target)
        return session.sparkContext.parallelize(Array(rf, lr, gbt, mpc)).collect()
    }

    def predict(data: Dataset[Observation], ensamble: Array[Transformer],
    session: SparkSession): RDD[((Double, DenseVector), Double)] = {
        import session.implicits._
        var collectedPredictions = ensamble.map(x => x.transform(data))
            .reduce{ (a: DataFrame, b: DataFrame) =>
                a.select("label", "features", "prediction").union(b.select("label", "features", "prediction"))
            }.map { row =>
                ((row.getDouble(0), row.getAs[DenseVector](1)), Array[Double](row.getDouble(2)))
            }.rdd.groupByKey().map{x =>
                var modeValue = mode(x._2, session)
                (x._1, modeValue)
            }
        return collectedPredictions
    }

    def computeEnsembleError(predictions: RDD[((Double, DenseVector), Double)],
        data: Dataset[Observation]) {
        var wrongPredictions = predictions.map{x =>
                var label = x._1._1
                var predicted = x._2
                if (predicted != label) {
                    1
                } else {
                    0
                }
            }.reduce(_+_)
        println("ENSAMBLE approach Test Error: " + wrongPredictions.toDouble/data.count())
    }

    def mode(data: Iterable[Array[Double]], session: SparkSession): Double = {
        var collected = data.map(x => x).reduce((a, b) => a ++ b)
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
