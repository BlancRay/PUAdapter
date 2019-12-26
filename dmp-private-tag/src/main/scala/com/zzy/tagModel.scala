package com.zzy

import java.io.FileInputStream
import java.util
import java.util.Properties

import com.fasterxml.jackson.databind.ObjectMapper
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.{Logger, LoggerFactory}

object tagModel {
    val prop = new Properties()
    val modelIdMap = new util.HashMap[String, String]()
    val LOG: Logger = LoggerFactory.getLogger("dmp-private-tag")
    val mapper = new ObjectMapper()

    /**
      * 根据参数model_id读取数据、训练模型、输出特征重要性、保存训练好的模型及参数
      *
      * @param args Array[String] 模型id
      */
    def main(args: Array[String]): Unit = {
        LOG.info("调用算法训练模型")

        //设置hadoop目录
        //        System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
        //        val modelid = "6"
        //        val sc = new SparkContext(new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]"))
        //        prop.load(new FileInputStream("E:\\xulei\\IdeaProjects\\dmp-private-tag\\model.properties"))

        val modelid = args(0)
        val sc = new SparkContext(new SparkConf().setAppName("dmp-private-tag"))
        //
        modelIdMap.put("key4token", "dmp")
        modelIdMap.put("modelId", modelid)

        try {
            prop.load(new FileInputStream(args(1)))
            tool.log("调用算法训练模型开始", "1")
            tool.log("读取模型参数", "1")
            val modelinfo = mapper.readTree(tool.postDataToURL(prop.getProperty("model_info"), modelIdMap)).get("outBean")
            LOG.info("模型信息:" + modelinfo)
            val algo_args = modelinfo.get("algorithmArgs")
            val model_args = modelinfo.get("modelArgs")

            //      val categoryInfo = tool.getCategoryInfo(modelid)
            val (attributeInfo, nominalInfo, attributeIndex, traitIDIndex) = tool.getAttributeInfo(modelid)
            //read p u data
            tool.log("读取种子人群", "1")

            val (_, p) = tool.read_convert(modelid, "P", sc, attributeInfo, traitIDIndex)
            LOG.info("P_SOURCE数量:" + p.count().toString)

            tool.log("读取训练数据", "1")
            val (_, u) = tool.read_convert(modelid, "U", sc, attributeInfo, traitIDIndex)
            LOG.info("U_SOURCE数量:" + u.count().toString)
            LOG.info("数据读取完成")

            //train model use p and u,return arg c and model
            LOG.info("开始训练模型")
            val (estC, model) = fit.fit(p, u, algo_args, nominalInfo.toMap)

            LOG.info("模型训练完成，正在保存模型")

            tool.save(model, estC, modelinfo, modelid, sc)

            tool.log("模型保存完成，开始计算模型影响因子", "1")
            LOG.info("开始计算模型影响因子")

            //            val feature_importance = importance.importance(model.trees, attributeInfo.size)
            val feature_importance = importance.featureImportances(model.trees, attributeInfo.size)

            tool.log("模型影响因子计算完成，正在保存...", "1")

            //setInfluenceFacByModelId
            val importance_mapArray = mapper.createArrayNode()
            feature_importance.keys.foreach { i => //index from 1
                val json_obj = mapper.createObjectNode()
                json_obj.put("traitId", attributeIndex(i - 1))
                if (feature_importance(i).isNaN)
                    json_obj.put("strength", "0.0")
                else
                    json_obj.put("strength", feature_importance(i).toString)
                importance_mapArray.add(json_obj)
            }

            LOG.info("写入模型影响因子")

            tool.postArrayToURL(prop.getProperty("influence"), importance_mapArray)

            tool.log("模型影响因子保存完成", "1")
            //            LOG.info("importance JSONArray"+importance_mapArray.toString)
            LOG.info("排序后feature_importance" + feature_importance.toList.sortBy(_._2).toString)
            LOG.info("开始计算相似度")

            val proba = model_test.evaluate(model, estC, modelid, sc, attributeInfo, traitIDIndex).collect()

            LOG.info(proba.toList.toString())
            tool.log("生成相似度与个体数量关系", "1")

            val similar = similarity.statistics(proba, model_args.get("delta").asDouble, modelinfo.get("sigma").asDouble).toMap

            tool.log("相似度与个体数量关系计算完成，正在保存...", "1")
            LOG.info("相似度计算完成，写入数据库中")

            //callBackSimilar
            val similarity_mapArray = mapper.createArrayNode()
            similar.keys.foreach { i =>
                val json_obj = mapper.createObjectNode()
                json_obj.put("similar", i.toString)
                json_obj.put("num", similar(i).toString)
                similarity_mapArray.add(json_obj)
            }
            // todo 数据量较大，需要分批写入
            tool.postArrayToURL(prop.getProperty("similar"), similarity_mapArray)
            tool.log("相似度与个体数量关系保存完成", "1")

            tool.log("模型训练完成", "2")
            LOG.info("模型训练完成\n\n\n\n")
        } catch {
            case e: Throwable =>
                tool.log("模型训练错误", "-1")
                LOG.error("模型训练错误\n" + e.toString)
                LOG.error(e.getMessage)
                LOG.error(e.getStackTrace.toString)
        } finally
            sc.stop()
    }
}
