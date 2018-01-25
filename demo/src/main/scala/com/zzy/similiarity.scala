package com.zzy

import scala.collection.mutable

protected object similiarity {
  /**
    * 统计不同相似度下人群数量占比
    *
    * @param proba     Array[Double] 个体的相似度
    * @param threshold Double 相似度变化率
    * @param ratio     Double 测试人群在数据库中占比
    * @return similarity mutable.Map[Double, Int]() (相似度,个体数量)
    */
  def statistics(proba: Array[Double], threshold: Double, ratio: Double): mutable.Map[Double, Int] = {
    val sn: Int = (1 / threshold).toInt
    val similarity = mutable.Map[Double, Int]()

    //    val Array_threshold: Array[Double] = new Array[Double](sn + 1)
    //    val Array_nbperson: Array[Double] = new Array[Double](sn + 1)
    var tmp: Double = 0.0
    for (i <- 0 until sn) {
      if (tmp == 0.0) {
        similarity.put(tmp, (proba.length / ratio).toInt)
      } else {
        var nbperson: Int = 0
        for (j <- proba) {
          if (j >= tmp) {
            nbperson = nbperson + 1
          }
        }
        similarity.put(tmp, (nbperson /ratio).toInt)
      }
      tmp = threshold + tmp
    }
    similarity
  }
}
