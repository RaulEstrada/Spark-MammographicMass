package main.scala

object Driver {
    def main(args: Array[String]) {
        if (args.length < 1) {
            println("MammographicMass Driver usage: <file>")
            System.exit(-1)
        }
    }
}
