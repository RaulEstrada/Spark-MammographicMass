package main.scala.preprocessing

import main.scala.schema.Observation
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.rdd.RDD

object DataPreProcessor {
    def preprocess(session: SparkSession, filePath: String) {
        import session.implicits._
        val observations = session.read.option("header", "true").option("inferSchema", "true")
            .csv(filePath).as[Observation]
        println("Read file with " + observations.count() + " observations")
        printMissingValues(session, observations)
    }

    def printMissingValues(session: SparkSession, data: Dataset[Observation]) {
        println("Missing values:")
        getColumnsMissingValues(session, data)
            .foreach{x => println("\t" + x)}
    }

    def getColumnsMissingValues(session: SparkSession, data: Dataset[Observation]):
    RDD[(String, Integer)] = {
        import session.implicits._
        return data.flatMap(x => x.toArray()).rdd
            .reduceByKey(_+_)
            .filter(x => x._2 > 0)
    }
}
