package main.scala.ml

import org.apache.spark.sql.Dataset
import main.scala.schema.Observation

object Ensembler {
    def ensemble(data: Dataset[Observation], target: String) {
        val rf = RF.generate(data, target)
        val lr = LR.generate(data, target)
        val gbt = GBT.generate(data, target)
        val mpc = MPC.generate(data, target)
    }
}
