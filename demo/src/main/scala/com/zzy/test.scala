package com.zzy

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

object test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]")//本地测试，上线时修改
    val sc = new SparkContext(conf)
    sc.textFile("").collect()
    val n= Array[String]("id1,1,123 12","id2,0,2 21","id3,1,3 31")
    val label=new Array[Int](n.length)
    val features=new Array[DenseVector](n.length)
    val lp=new Array[LabeledPoint](n.length)
    for (i <-  n.indices){
      val splited = n(i).split(",")
      val feature = splited(2).split(" ")
      val array = new Array[Double](feature(0).length-1)
      for (j<- array.indices){
        array(j)= feature(j).toDouble
      }

      val dv: Vector = Vectors.dense(array)
      println(feature(0).toDouble)
      lp(i) = LabeledPoint(splited(1).toDouble,dv)
    }
    val rdd = sc.makeRDD(lp)
    println(rdd.map(_.features).take(1).toVector)

  }
}