package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.functions.min
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"

    val obsDF: DataFrame = sc.textFile(filePath).toDF("line")

    val pipeline = this.getPipeline(obsDF)
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    val cleanDF = pipelineModel.transform(obsDF)

    val myLR= new MyLinearRegressionImpl()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
    val lrStage = myLR.fit(cleanDF)
    val predictions = lrStage.transform(cleanDF)
    predictions.select("label", "prediction").take(5).foreach(println)

    val summary = lrStage.trainingError.last
    println("RMSE: %f".format(summary))
    //print rmse of our model
    //do prediction - print first k
  }
  private def getPipeline(df: DataFrame): Pipeline = {
    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("line")
      .setOutputCol("tokens")
      .setPattern(",")
    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("vector")
    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("year").setIndices(Array(0))
    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF((vec) => vec(0)).setInputCol("year").setOutputCol("yeard")
    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    val minYear = v2d.transform(
      lSlicer.transform(
        arr2Vect.transform(
          regexTokenizer.transform(df)))).agg(min("yeard")).head.getDouble(0)
    val lShifter = new DoubleUDF(_ - minYear).setInputCol("yeard").setOutputCol("label")
    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("features").setIndices((1 to 3).toArray)
    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))

    pipeline
  }
}