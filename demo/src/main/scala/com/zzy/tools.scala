package com.zzy

import java.io.{File, PrintWriter}
import Array._
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.hadoop.hbase.client.Result
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat

import scala.collection.mutable

class tools{
  /**
    * 将一个数组保存到指定文件中
    * @param array Array[Double] 数组
    * @param dir String 文件地址
    */
  def Array2file(array:Array[Double], dir:String){
    val writer = new PrintWriter(new File(dir))//模型概率文件
    for (i <- array.indices)
      if(array(i)>1)
        writer.println(1)
      else
        writer.println(array(i))
    writer.close()
  }

  /**
    * 将两个长度相等的数组保存为文件
    * @param array1 Array[Double] 数组1
    * @param array2 Array[Double] 数组2
    * @param dir String 文件保存地址
    */
  def Arrays2file(array1:Array[Double], array2:Array[Double], dir:String){
    val writer = new PrintWriter(new File(dir))
    for (i <- array2.indices)
      writer.println(array1(i) + "," + array2(i))
    writer.close()
  }

  /**
    * 将两个长度相等的数组保存为文件
    * @param array1 Array[String] 数组1
    * @param array2 Array[Double] 数组2
    * @param dir String 文件保存地址
    */
  def Arrays2file(array1:Array[String], array2:Array[Double], dir:String){
    val writer = new PrintWriter(new File(dir))
    for (i <- array2.indices)
      writer.println(array1(i) + "," + array2(i))
    writer.close()
  }

  /**
    * 将一个数组保存到指定SQL表中
    * @param array Array[Double] 数组
    * @param dir String SQL参数
    */
  def Array2SQL(array:Array[Double],dir:String){
    //todo
  }

  /**
    * 将两个等长数组保存到指定SQL表中
    * @param array1 Array[Double] 数组1
    * @param array2 Array[Double] 数组2
    * @param dir String 文件地址
    */
  def Arrays2SQL(array1:Array[Double],array2:Array[Double],dir:String){
    //todo
  }

  /**
    * 将两个等长数组保存到指定SQL表中
    * @param array1 Array[String] 数组1
    * @param array2 Array[Double] 数组2
    * @param dir String SQL参数
    */
  def Arrays2SQL(array1:Array[String],array2:Array[Double],dir:String){
    //todo
  }
  /**
    * 从指定的路径中读取数据，数据格式为org.apache.spark.mllib.regression.LabeledPoint
    * @param dir 文件地址
    * @param sc SparkContext
    * @return (RDD[String],RDD[LabeledPoint])(样本ID,标签及特征)
    */
  def readData(dir: String, sc: SparkContext): (RDD[String],RDD[LabeledPoint]) = {
    val n = sc.textFile(dir).collect()
    val gid = new Array[String](n.length)
    val lp = new Array[LabeledPoint](n.length)
    for (i <- n.indices) {
      val split = n(i).split(",")
      gid(i) = split(0)
      val feature = split(2).split(" ")
      val array = new Array[Double](feature.length)
      for (j <- array.indices) {
        array(j) = feature(j).toDouble
      }
      val dv: Vector = Vectors.dense(array)
      lp(i) = LabeledPoint(split(1).toDouble, dv)
    }
    val lpRDD = sc.makeRDD(lp)
    val gidRDD = sc.makeRDD(gid)
    (gidRDD, lpRDD)
  }

  /**
    * 保存HashMap数据到指定SQL表中
    * @param list mutable.HashMap[Int, Double] HashMap
    * @param dir String 文件地址
    */
  def List2file(list:List[(Int, Double)], dir:String): Unit ={
    val writer = new PrintWriter(new File(dir)) //模型特征重要性文件
    list.foreach(e =>
      writer.println(e._1 + "," + e._2)
    )
    writer.close()
  }

  /**
    * 保存HashMap数据到指定SQL表中
    * @param list mutable.HashMap[Int, Double] HashMap
    * @param dir String SQL参数
    */
  def List2SQL(list:List[(Int, Double)],dir:String): Unit ={
    //todo
  }

  /**
    * 计算模型对数据的预测概率
    * @param points RDD[LabeledPoint]
    * @param model RandomForestModel
    * @return RDD[Double] 预测为正例的概率
    */
  def predict(points: RDD[LabeledPoint], model: RandomForestModel): RDD[Double] = {
    val numTrees = model.trees.length
    val trees = points.sparkContext.broadcast(model.trees)
    points.map { point =>
      trees.value.map(_.predict(point.features)).sum / numTrees
    }
  }

  /**
    *
    * @param Type String 数据来源，P/U/N
    * @param sc SparkContext
    * @return (RDD[String],RDD[LabeledPoint]) (GID,数据)
    */
  def convert(Type:String,sc:SparkContext):(RDD[String],RDD[LabeledPoint])= {
    val conf = HBaseConfiguration.create()
    if(Type.contains("P"))
      conf.set(TableInputFormat.INPUT_TABLE, "P_SOURCE")
    else if(Type.contains("U"))
      conf.set(TableInputFormat.INPUT_TABLE, "U_SOURCE")
    else if(Type.contains("N"))
      conf.set(TableInputFormat.INPUT_TABLE, "N_SOURCE" )

    // TODO: 根据Type读取对应数据
    val data_source = sc.newAPIHadoopRDD(conf, classOf[TableInputFormat],classOf[org.apache.hadoop.hbase.io.ImmutableBytesWritable],classOf[org.apache.hadoop.hbase.client.Result]).collect()

    val gid = new Array[String](data_source.length)
    val lp = new Array[LabeledPoint](data_source.length)

    for (i <- data_source.indices) {
      gid(i) = "id1"// TODO:从SOURCE表中读取的某个GID的ROWKEY
      val hbaseresult = "1:1,2:123,5:1,6:1234"// TODO:从SOURCE表中读取的某个GID的column标签集合
      val GID_TAG_SET = hbaseresult.split(" ")
      val GID_TAG_split = new scala.collection.mutable.HashMap[Int, Double]()
      GID_TAG_SET.foreach(each => {
        val tag_split = each.split(":")
        GID_TAG_split.put(tag_split(0).toInt, tag_split(1).toDouble)
      })
      val factors = range(0,10)// TODO: 从MODEL_TAG_INDEX表中读取modelID的所有标签
      val feature = new Array[Double](factors.length)
      factors.foreach(each => {
        if (GID_TAG_split.contains(each)) {
          feature(each) = GID_TAG_split(each)
        } else {
          feature(each) = 0
        }
      }
      )
      val dv: Vector = Vectors.dense(feature)
      if (Type.contains("P"))
        lp(i) = LabeledPoint(1, dv)
      else
        lp(i) = LabeledPoint(0, dv)
    }
    (sc.makeRDD(gid),sc.makeRDD(lp))
  }
}