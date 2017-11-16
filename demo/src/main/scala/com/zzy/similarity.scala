package com.zzy

import java.io._
import java.util.Properties

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

object similarity {
  val prop: Properties = new Properties()

  /**
    * 根据传入的模型id，相似度计算粒度，读取在测试数据上得到的概率文件，将统计好的相似度阈值-人群数量占比输出到MySQL表中
    *
    * @param args 模型id
    */
  def main(args: Array[String]): Unit = {
    /**
      * 读取模型配置文件
      */
    println("read properties")
    val in: InputStream = new BufferedInputStream(new FileInputStream(args(0) + ".properties"))
    prop.load(in)
    val conf = new SparkConf().setAppName("Prob2AS").setMaster("local[1]")
    val sc = new SparkContext(conf)
    println("Reading file")
    val RDD_prob = sc.textFile(prop.getProperty("prob_dir"), sc.defaultMinPartitions).map(x => x.toDouble)
    val Array_prob = RDD_prob.map(x => x.formatted("%.6f").toDouble).collect()

    println("doing statistics")
    val threshold: Double = prop.getProperty("threshold").toDouble
    val (array_as_threshold, array_as_nbperson) = statistics(Array_prob, threshold)

    val tool = new tools()
    tool.Arrays2file(array_as_threshold, array_as_nbperson, prop.getProperty("similarity"))
    //    tool.Arrays2SQL(array_as_threshold, array_as_nbperson, prop.getProperty("similarity"))
    println("finished")
    sc.stop()
  }

  /**
    * 统计不同相似度下人群数量占比
    *
    * @param proba     Array[Double] 个体的相似度
    * @param threshold Double 相似度变化率
    * @return (Array_threshold,Array_nbperson) (Array[Double],Array[Double]) (相似度,个体数量占比)
    */
  private def statistics(proba: Array[Double], threshold: Double) = {
    val sn: Int = (1 / threshold).toInt

    val Array_threshold: Array[Double] = new Array[Double](sn + 1)
    val Array_nbperson: Array[Double] = new Array[Double](sn + 1)
    var tmp: Double = 0.0
    for (i <- 0 until sn) {
      if (tmp == 0.0) {
        Array_threshold(i) = tmp
        Array_nbperson(i) = 1
      } else {
        var nbperson: Int = 0
        for (j <- proba) {
          if (j >= tmp) {
            nbperson = nbperson + 1
          }
        }
        Array_threshold(i) = tmp
        Array_nbperson(i) = 1.0 * nbperson / proba.length
      }
      tmp = threshold + tmp
    }
    (Array_threshold, Array_nbperson)
  }
}