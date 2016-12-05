package main.scala

import org.apache.spark.sql.SparkSession
import main.scala.preprocessing.DataPreProcessor

object Driver {
    def main(args: Array[String]) {
        if (args.length < 1) {
            println("MammographicMass Driver usage: <file>")
            System.exit(-1)
        }
        // Entry point of Spark SQL
        val session = SparkSession.builder().getOrCreate();
        DataPreProcessor.preprocess(session, args(0))
    }
}
