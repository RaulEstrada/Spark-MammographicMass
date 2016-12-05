package main.scala.schema

case class Observation(val BIRADS: Option[Int], val Age: Option[Int],
    val Shape: Option[Int], val Margin: Option[Int], val Density: Option[Int],
    val Severity: Option[Int]) {

}
