package main.scala.preprocessing

import main.scala.schema.Observation
import org.apache.spark.sql.SparkSession

object DataPreProcessor {
    def preprocess(session: SparkSession, filePath: String) {
        import session.implicits._
        val file = session.read.option("header", "true").option("inferSchema", "true")
            .csv(filePath).as[Observation]
        println("Read file with " + file.count() + " observations")
    }
}
