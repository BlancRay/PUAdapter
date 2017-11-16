package com.zzy

import java.io._
import java.util.Properties

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

import scala.io.Source

object RandomForestTest {
  val prop: Properties = new Properties()

  /**
    * 根据参数获取模型参数，读取测试数据、模型及参数c、输出预测概率
    *
    * @param model_id String 模型id
    * @param data     RDD[LabeledPoint] 测试数据
    * @return (gid,proba) (Array[String],Array[Double])
    */
  def evaluate(model_id: String, data: RDD[LabeledPoint], sc: SparkContext): RDD[Double] = {
    val tool = new tools()
    println("read properties")
    val in: InputStream = new BufferedInputStream(new FileInputStream(model_id + ".properties"))
    prop.load(in)
    println("start")
    // $example on$
    // Load and parse the data file.
    println("load file")

    println("testing")
    val model = RandomForestModel.load(sc, prop.getProperty("model_dir"))
    // Evaluate model on test instances and compute test error
    // Save and load model
    val prediction = tool.predict(data, model)
    val estC = Source.fromFile(prop.getProperty("model_c"), "UTF-8").getLines().indexOf(0)
    val proba = prediction.map(_ / estC.toDouble)

    println("finished")
    sc.stop()
    proba
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RandomForestClassificationTest").setMaster("local[4]")
    //本地测试，上线时修改
    val sc = new SparkContext(conf)
    val tool = new tools()
    val (n_gid, n) = tool.readData(prop.getProperty("n_dir"), sc)
    //测试数据
    val proba = evaluate(args(0), n, sc)
    tool.Arrays2file(n_gid.collect(), proba.collect(), prop.getProperty("prob_dir"))
    //    tool.Arrays2SQL(result,prop.getProperty("prob_dir"))
  }
}
