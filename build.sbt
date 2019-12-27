name := "spark-rec"

version := "0.1"

scalaVersion := "2.11.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.0"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.3.0" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-streaming-kafka" % "1.6.0"

libraryDependencies += "org.mongodb" %% "casbah" % "3.1.1"

libraryDependencies += "org.jblas" % "jblas" % "1.2.4"

libraryDependencies += "net.jpountz.lz4" % "lz4" % "1.3.0"
