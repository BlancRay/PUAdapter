package com.zzy

import com.zzy.tagModel.LOG
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

import scala.collection.mutable

protected object model_test {
    /**
      * 根据参数获取模型参数，读取测试数据、模型及参数c、输出预测概率
      *
      * @param model RandomForestModel 模型
      * @param estC  Double c
      * @return RDD[Double] 预测概率
      */
    def evaluate(model: RandomForestModel, estC: Double, modelid: String, sc: SparkContext, Features: mutable.Map[Int, Int], traitIDIndex: mutable.Map[String, Int]): RDD[Double] = {
        tool.log("生成模型测试数据", "1")
        val (_, test_data) = tool.read_convert(modelid, "N", sc, Features, traitIDIndex)
        LOG.info("N_SOURCE数量:" + test_data.count().toString)
        tool.log("模型测试中", "1")
        val prediction = tool.predict(test_data, model)
        val proba = prediction.map(_ / estC)
        tool.log("模型测试完成", "1")
        proba
    }
}
