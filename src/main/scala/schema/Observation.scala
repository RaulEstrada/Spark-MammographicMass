package main.scala.schema

case class Observation(val BIRADS: Option[Int], val Age: Option[Int],
    val Shape: Option[Int], val Margin: Option[Int], val Density: Option[Int],
    val Severity: Option[Int]) {
    // BIRADS: Assessment ranging from 1 (definitely benign) to 5 (highly
    // suggestive of malignancy)

    //1. BI-RADS assessment: 1 to 5 (ordinal, non-predictive!)
    //2. Age: patient's age in years (integer)
    //3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
    //4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
    //5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
    //6. Severity: benign=0 or malignant=1 (binominal, goal field!)

    def toArray(): Array[(String, Integer)] = {
        return Array[(String, Integer)](
            ("BIRADS", if (BIRADS.isEmpty) {1} else {0}),
            ("Age", if (Age.isEmpty) {1} else {0}),
            ("Shape", if (Shape.isEmpty) {1} else {0}),
            ("Margin", if (Margin.isEmpty) {1} else {0}),
            ("Density", if (Density.isEmpty) {1} else {0}),
            ("Severity", if (Severity.isEmpty) {1} else {0})
        )
    }
}

object Observation {
    def getFeaturesArray(target: String): Array[String] = {
        target match {
        case "Margin" => return Array[String]("BIRADS", "Age")
        case "Shape" => return Array[String]("BIRADS", "Age", "Margin")
        case "Density" => return Array[String]("BIRADS", "Age", "Margin", "Shape")
        case "Severity" => return Array[String]("BIRADS", "Age", "Margin", "Shape")
        }
    }
}
