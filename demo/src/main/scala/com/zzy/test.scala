package com.zzy

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import net.sf.json.{JSONArray, JSONObject}

import scala.util.parsing.json.JSON

object test {
  def main(args: Array[String]): Unit = {
    println((5/0.1).toInt)
    sys.exit()
    val str = """{"outBean":{"model_id":9,"model_name":"模型001","tagId":3780,"instensity_begin":0.6,"instensity_end":0.8,"recency_begin":"1512701801250","recency_end":"1512976540000","status":2,"algorithmId":1,"except_tags":"3776,3777,3778","train_time":"1512976540000","algorithm_args":{"maxDepth":"32","featureSubsetStrategy":"auto","categoricalFeaturesInfo":"1","maxBins":"100","numTrees":"512","numClasses":"2"," holdOutRatio":"0.1","impurity":"gini"},"model_dir":"","model_args":{"delta":"0.0001","alpha":"1","beta":"2"},"tmp_param":"","nbp":"9","sigma":"null"},"code":"00","msg":"success"}"""
    val jsobj = JSONObject.fromObject(JSONObject.fromObject(str).get("outBean")).getString("algorithm_args")
    println(JSONObject.fromObject(jsobj))
    val hold_out_ratio = JSONObject.fromObject(jsobj).getString(" holdOutRatio").toDouble
    print(hold_out_ratio)
    sys.exit()
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