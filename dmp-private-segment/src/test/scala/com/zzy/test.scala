package com.zzy

import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object test {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]")
        //本地测试，上线时修改
        val sc = new SparkContext(conf)
        //    val modelinfo = JSONObject.fromObject(tool.postDataToURL(prop.getProperty("model_info"), modelIdMap)).get("outBean").toString
        //    RandomForestModel.load(sc,prop.getProperty("hdfs_dir") + JSONObject.fromObject(modelinfo).get("model_dir") + "/" + 16)
        //    sys.exit()
        //    val str = """{"outBean":{"model_id":9,"model_name":"模型001","tagId":3780,"instensity_begin":0.6,"instensity_end":0.8,"recency_begin":"1512701801250","recency_end":"1512976540000","status":2,"algorithmId":1,"except_tags":"3776,3777,3778","train_time":"1512976540000","algorithm_args":{"maxDepth":"32","featureSubsetStrategy":"auto","categoricalFeaturesInfo":"1","maxBins":"100","numTrees":"512","numClasses":"2"," holdOutRatio":"0.1","impurity":"gini"},"model_dir":"","model_args":{"delta":"0.0001","alpha":"1","beta":"2"},"tmp_param":"","nbp":"9","sigma":"null"},"code":"00","msg":"success"}"""
        //    val jsobj = JSONObject.fromObject(JSONObject.fromObject(str).get("outBean")).getString("algorithm_args")
        //    println(JSONObject.fromObject(jsobj))
        //    val hold_out_ratio = JSONObject.fromObject(jsobj).getString(" holdOutRatio").toDouble
        //    print(hold_out_ratio)
        //    sys.exit()
        //    sc.textFile("").collect()
        val n = Array[String]("id1,1,123 12", "id2,0,2 21", "id3,1,3 31")
        val drdd = sc.makeRDD(n)
        val l = drdd.count().toInt
        val label = new Array[Int](l)
        val features = new Array[DenseVector](l)
        val lp = new Array[LabeledPoint](l)
        val lprdd = drdd.map { each =>
            val splited = each.split(",")
            val feature = splited(2).split(" ")
            val array = new Array[Double](feature.length)
            for (j <- array.indices) {
                array(j) = feature(j).toDouble
            }

            val dv: Vector = Vectors.dense(array)
            //            println(feature(0).toDouble)
            (splited(0), LabeledPoint(splited(1).toDouble, dv))
        }
        val gid = lprdd.map(_._1)
        val lps = lprdd.map(_._2)
        println(gid.take(3).toVector)

        //    for (i <-  n.indices){
        //      val splited = n(i).split(",")
        //      val feature = splited(2).split(" ")
        //      val array = new Array[Double](feature(0).length-1)
        //      for (j<- array.indices){
        //        array(j)= feature(j).toDouble
        //      }
        //
        //      val dv: Vector = Vectors.dense(array)
        //      println(feature(0).toDouble)
        //      lp(i) = LabeledPoint(splited(1).toDouble,dv)
        //    }
        //    val rdd = sc.makeRDD(lp)
        println(lps.take(9).toVector)

    }
}