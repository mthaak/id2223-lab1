package se.kth.spark.lab1.task1

import se.kth.spark.lab1._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

case class Song(year: Double, f1: Double, f2: Double, f3: Double)

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    //    val rawDF = ???

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    // delimiter = ,
    // 12 features
    // datatypes = all doubles
    rdd.take(5).foreach(println)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(_.split(","))

    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map((line: Array[String]) => Song(line(0).toDouble, line(1).toDouble, line(2).toDouble, line(3).toDouble))

    //Step4: convert your rdd into a datafram
    val songsDf = songsRdd.toDF()

    // Check DF conversion worked
    songsDf.take(5).foreach(a => println(a.toString()))

    import org.apache.spark.sql.functions._

    // Questions
    songsDf.createOrReplaceTempView("songs")
    //    1. How many songs there are in the DataFrame?
    println("%d songs".format(songsDf.count()))
    sparkSession.sql("SELECT count(*) as num_songs FROM songs").show()
    //    2. How many songs were released between the years 1998 and 2000?
    println("%d songs released between the 1998 and 2000".format(songsDf.filter($"year" >= 1998 && $"year" < 2000).count()))
    sparkSession.sql("SELECT count(*) as num_songs FROM songs WHERE 1998 <= year AND year < 2000").show()
    //    3. What is the min, max and mean value of the year column?
    println("%f min year".format(songsDf.agg(min("year")).head.getDouble(0)))
    println("%f max year".format(songsDf.agg(max("year")).head.getDouble(0)))
    println("%f mean year".format(songsDf.agg(mean("year")).head.getDouble(0)))
    sparkSession.sql("SELECT min(year) FROM songs").show()
    sparkSession.sql("SELECT max(year) FROM songs").show()
    sparkSession.sql("SELECT avg(year) FROM songs").show()
    //    4. Show the number of songs per year between the years 2000 and 2010?
    println("%d songs per year between 2000 and 2010".format(songsDf.filter($"year" >= 2000 && $"year" < 2010).groupBy("year").agg(count("f1")).head().getInt(0)))
    sparkSession.sql("SELECT year, count(f1) FROM songs WHERE 2000 <= year AND year < 2010 GROUP BY year").show()
  }
}