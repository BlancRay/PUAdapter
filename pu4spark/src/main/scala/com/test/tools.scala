package com.test

import java.io.File

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
            val feature = new Array[Double](features.length)
            for (j <- feature.indices) {
                feature(j) = features(j).toDouble
            }
            val dv: Vector = Vectors.dense(feature)
            if (dtype == "P")
                lp.update(i, LabeledPoint(1, dv))
            else if (dtype == "U")
                lp.update(i, LabeledPoint(0, dv))
            else if (dtype == "N")
                if (i < 700) lp.update(i, LabeledPoint(1, dv)) else lp.update(i, LabeledPoint(0, dv))
            i += 1
        }
        sc.makeRDD(lp)
    }

    def getAttributeInfo(modelid: String): (mutable.Map[Int, Int], mutable.Map[Int, Int]) = {
        val json = "{" + Source.fromFile("model/test/test/AttrbuiteJSON.txt").getLines().next() + "}"
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

    def dirDel(path: File) {
        if (!path.exists())
            return
        else if (path.isFile) {
            path.delete()
            pu4sparktest.logger.warn(path + ":  文件被删除")
            return
        }
        val file: Array[File] = path.listFiles()
        for (d <- file) {
            dirDel(d)
        }
        path.delete()
        pu4sparktest.logger.warn(path + ":  目录被删除")
    }
}
