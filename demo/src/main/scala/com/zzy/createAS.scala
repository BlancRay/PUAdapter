package com.zzy

import java.io._
import java.util.Properties

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

object createAS {
  val prop: Properties = new Properties()

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RandomForestClassificationTest").setMaster("local[4]")
    //本地测试，上线时修改
    val sc = new SparkContext(conf)
    val in: InputStream = new BufferedInputStream(new FileInputStream(args(0) + ".properties"))
    prop.load(in)
    println("read properties")
    val tool = new tools()
    val (n_gid, n) = tool.readData(prop.getProperty("n_dir"), sc)
    //测试数据
    val gid = addAS(args(0), n_gid, n, sc)
    //todo add gid into GID_GROUP
  }

  def addAS(model_id: String, gid: RDD[String], data: RDD[LabeledPoint], sc: SparkContext): RDD[String] = {
    val in: InputStream = new BufferedInputStream(new FileInputStream(model_id + ".properties"))
    prop.load(in)
    val threshold = Source.fromFile(prop.getProperty("threshold_dir"), "UTF-8").getLines().indexOf(0).toDouble //载入模型相似度threshold
    println("start")
    val proba = RandomForestTest.evaluate(model_id, data, sc)
    val result = proba.filter(_ > threshold).collect()
    val addgid = new Array[String](gid.count().toInt)
    val gids = gid.collect()
    for (i <- 0 until result.length - 1) {
      if (result(i) >= threshold) {
        addgid(i) = gids(i)
      }
    }
    sc.makeRDD(addgid)
  }
}
