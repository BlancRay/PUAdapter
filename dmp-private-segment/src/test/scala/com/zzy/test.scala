package com.zzy

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

object test {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]").set("spark.sql.warehouse.dir", "file:///E:/xulei/zhiziyun/")
        //本地测试，上线时修改
        val sc = new SparkContext(conf)
        //        val model = RandomForestModel.load(sc, "./dmp-private-segment/34")
        //根据Type读取对应数据
        val data_source = sc.textFile("D:/1.txt")

        val factors = Array.range(0, 1000)
        val result = data_source.map { each =>
            val hbaseresult = each
            val GID_TAG_split = new mutable.HashMap[Int, Double]()
            if (hbaseresult != "") {
                val GID_TAG_SET = hbaseresult.split(";")
                var is_time = true
                GID_TAG_SET.foreach { each =>
                    val tag_split = each.split(":")
                    if (is_time) {
                        GID_TAG_split.put(tag_split(0).toInt, tag_split(1).toDouble / 3600000)
                        is_time = false
                    } else {
                        GID_TAG_split.put(tag_split(0).toInt, tag_split(1).toDouble)
                        is_time = true
                    }
                }
            }
            val feature = new Array[Double](factors.length)
            factors.foreach { f_each =>
                if (GID_TAG_split.contains(f_each + 1)) {
                    feature(f_each) = GID_TAG_split(f_each + 1)
                } else {
                    feature(f_each) = 0
                }
            }
            val dv: Vector = Vectors.dense(feature)
            var lp: LabeledPoint = null
            lp = LabeledPoint(1, dv)
            (1, lp)

        }
        println(result.take(1).toList)
        //        (result.map(_._1), result.map(_._2))
    }
}