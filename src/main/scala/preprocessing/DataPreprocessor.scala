package main.scala.preprocessing

import main.scala.schema.Observation
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.rdd.RDD

object DataPreProcessor {
    val deletionUpperLimit = 10

    def preprocess(session: SparkSession, filePath: String) {
        import session.implicits._
        var observations = session.read.option("header", "true").option("inferSchema", "true")
            .csv(filePath).as[Observation]
        println("Read file with " + observations.count() + " observations")
        printMissingValues(session, observations)
        println("Removing observations with columns whose number of NA is lower than: " + deletionUpperLimit)
        observations = handleDeletionNAs(session, observations)
        println("After deletion, dataset with size: " + observations.count())
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

    def handleDeletionNAs(session: SparkSession, data: Dataset[Observation]):
    Dataset[Observation] = {
        import session.implicits._
        val targetColumns = getColumnsMissingValues(session, data)
            .filter(x => x._2 < deletionUpperLimit)
            .map(x => x._1).collect()
        data.na.drop("any", targetColumns).as[Observation]
    }
}
