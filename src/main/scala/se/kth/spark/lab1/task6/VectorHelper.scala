package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.ml.linalg.Vectors

import scala.collection.mutable.ArrayBuffer

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    v1.toArray.zip(v2.toArray).map(x =>x._1 * x._2).sum
  }

  def dot(v: Vector, s: Double): Vector = {
    Vectors.dense(v.toArray.map(s * _))
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    Vectors.dense(v1.toArray.zip(v2.toArray).map({case(x,y)=>x+y}))
  }

  def fill(size: Int, fillVal: Double): Vector = {
    Vectors.dense((0 until size).map(x => fillVal).toArray)
  }
}