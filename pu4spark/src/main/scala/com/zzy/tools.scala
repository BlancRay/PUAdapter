package com.zzy

import net.sf.json.JSONObject
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.io.Source

object tools {
    def readData(sc: SparkContext, data: RDD[String], dtype: String): RDD[LabeledPoint] = {
        val lp = new Array[LabeledPoint](data.count().toInt)
        var i = 0
        data.collect().foreach { eachdata =>
            val features = eachdata.split(",")
            val feature = new Array[Double](features.length) //6868
            for (j <- feature.indices) {
                feature(j) = features(j).toDouble
            }
            val dv: Vector = Vectors.dense(feature)
            if (dtype == "P")
                lp.update(i, LabeledPoint(1, dv))
            else
                lp.update(i, LabeledPoint(0, dv))
            i += 1
        }
        sc.makeRDD(lp)
    }

    def getAttributeInfo(modelid: String): (mutable.Map[Int, Int], mutable.Map[Int, Int]) = {
        val json = "{" + Source.fromFile("E:\\xulei\\zhiziyun\\model\\test\\test\\AttrbuiteJSON.txt").getLines().next() + "}"
        val featureInfoJson = JSONObject.fromObject(json)
        val nominalInfo = mutable.Map[Int, Int]()
        val attributeInfo = mutable.Map[Int, Int]()
        for (i <- 0 until featureInfoJson.size()) {
            val a = featureInfoJson.getInt(i.toString)
            attributeInfo.put(i, a)
            if (a != 1)
                nominalInfo.put(i, a)
        }
        (attributeInfo, nominalInfo)
    }
}
