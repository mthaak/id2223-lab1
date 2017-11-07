package se.kth.spark.lab1.task2

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.min
import se.kth.spark.lab1._


object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext._
    import sqlContext.implicits._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF("line")

    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("line")
      .setOutputCol("tokens")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val songsTokenized = regexTokenizer.transform(rawDF)
    songsTokenized.take(5).foreach(println)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("vector")
    val songsTokenizedVector = arr2Vect.transform(songsTokenized)
    songsTokenizedVector.take(5).foreach(println)

    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("year")
    lSlicer.setIndices(Array(0))
    val lSliced = lSlicer.transform(songsTokenizedVector)
    lSliced.take(5).foreach(println)

    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF((vec) => vec(0)).setInputCol("year").setOutputCol("yeard")
    val yearAsDouble = v2d.transform(lSliced)
    yearAsDouble.take(5).foreach(println)

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    val minYear = yearAsDouble.agg(min("yeard")).head.getDouble(0)
    val lShifter = new DoubleUDF(_ - minYear).setInputCol("yeard").setOutputCol("label")
    val lShifted = lShifter.transform(yearAsDouble)
    lShifted.take(5).foreach(println)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("features")
    fSlicer.setIndices((1 to 3).toArray)
    val fSliced = fSlicer.transform(lShifted)
    fSliced.take(5).foreach(println)

    //Step 7.5: drop all but label and features
    //val dropper = new ColumnPruner(Set("line", "tokens", "vector", "year", "yeard")).setInputCol()
    //val cDropped = dropper.transform(sliced_features)
    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions
    val result = pipelineModel.transform(rawDF)
    result.take(5).foreach(println)

    //Step11: drop all columns from the dataframe other than label and features
    val final_ = result.select("label", "features")
    final_.take(5).foreach(println)
  }
}