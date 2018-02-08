package com.zzy

import java.io.{BufferedInputStream, FileInputStream}
import java.util
import java.util.Properties

import net.sf.json.{JSONArray, JSONObject}
import org.apache.spark.{SparkConf, SparkContext}
import org.jsoup.HttpStatusException
import org.slf4j.{Logger, LoggerFactory}

object rf {
  val prop = new Properties()
  val modelIdMap = new util.HashMap[String, String]()
  val LOG: Logger = LoggerFactory.getLogger("rf")

  /**
    * 根据参数model_id读取数据、训练模型、输出特征重要性、保存训练好的模型及参数
    *
    * @param args Array[String] 模型id
    */
  def main(args: Array[String]): Unit = {
    LOG.info("\n\n\n调用算法训练模型")

    //设置hadoop目录
    //    System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
    //    val modelid = "16"
    //    val sc = new SparkContext(new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]"))
    //    prop.load(this.getClass.getResourceAsStream("../model.properties"))

    val modelid = args(0)
    val sc = new SparkContext(new SparkConf().setAppName("RandomForestClassificationTrain"))
    prop.load(new BufferedInputStream(new FileInputStream(args(1))))

    tool.log(modelid, "调用算法训练模型开始", "1", prop.getProperty("log"))
    //read model info
    //    val params = new util.HashMap[String, String]()
    modelIdMap.put("key4token", "dmp")
    modelIdMap.put("modelId", modelid)
    try {
      tool.log(modelid, "读取模型参数", "1", prop.getProperty("log"))
      val modelinfo = JSONObject
        .fromObject(
          tool.postDataToURL(prop.getProperty("model_info"), modelIdMap))
        .get("outBean")
        .toString
      LOG.info("模型信息:" + modelinfo)
      val algo_args =
        JSONObject.fromObject(modelinfo).get("algorithm_args").toString
      val model_args =
        JSONObject.fromObject(modelinfo).get("model_args").toString

      val TagIndexInfo =
        tool.postDataToURL(prop.getProperty("tagindex"), modelIdMap)
      val TagArray =
        JSONArray.fromObject(JSONObject.fromObject(TagIndexInfo).get("result"))
      //read p u data
      tool.log(modelid, "读取种子人群", "1", prop.getProperty("log"))

      val (_, p) = tool.read_convert(modelid, "P_SOURCE", sc, TagArray.size())
      LOG.info("P_SOURCE数量:" + p.count().toString)

      tool.log(modelid, "读取训练数据", "1", prop.getProperty("log"))
      val (_, u) = tool.read_convert(modelid, "U_SOURCE", sc, TagArray.size())
      LOG.info("U_SOURCE数量:" + u.count().toString)
      LOG.info("数据读取完成")

      //train model use p and u,return arg c and model
      val (estC, model) = fit.fit(modelid, p, u, algo_args)

      LOG.info("模型训练完成，正在保存模型")
      tool.save(model, estC, modelinfo, modelid, sc)

      tool.log(modelid, "模型保存完成，开始计算模型影响因子", "1", prop.getProperty("log"))

      LOG.info("开始计算模型影响因子")
      //      val feature_importance = importance.importance(model.trees, TagArray.size())
      val feature_importance = importance.featureImportances(model.trees, TagArray.size())
      //setInfluenceFacByModelId
      val importance_mapArray = new JSONArray()
      feature_importance.keys.foreach { i =>
        val json_obj = new JSONObject()
        json_obj.put("tagIndex", i.toString)
        json_obj.put("instensity", feature_importance(i).toString)
        importance_mapArray.add(json_obj)
      }

      LOG.info(feature_importance.toList.sortBy(_._2).toString)
      LOG.info("写入模型影响因子")
      tool.postArrayToURL(modelid,
                          prop.getProperty("influence"),
                          importance_mapArray)

      LOG.info("开始计算相似度")
      val proba =
        model_test.evaluate(model, estC, modelid, sc, TagArray.size()).collect()
      tool.log(modelid, "生成相似度与个体数量关系", "1", prop.getProperty("log"))
      val similar = similarity
        .statistics(proba,
                    JSONObject.fromObject(model_args).getDouble("delta"),
                    JSONObject.fromObject(modelinfo).getDouble("sigma"))
        .toMap

      //callBackSimilar
      // todo 数据量较大，需要分批写入
      val similarity_mapArray = new JSONArray()
      similar.keys.foreach { i =>
        val json_obj = new JSONObject()
        json_obj.put("similar", i.toString)
        json_obj.put("num", similar(i).toString)
        similarity_mapArray.add(json_obj)
      }
      LOG.info("相似度计算完成，写入数据库中")
      tool.postArrayToURL(modelid,
                          prop.getProperty("similar"),
                          similarity_mapArray)
      tool.log(modelid, "模型训练完成", "2", prop.getProperty("log"))
      LOG.info("模型训练完成\n\n\n\n")
    } catch {
      case httpError: HttpStatusException =>
        LOG.error("服务器错误，Status code：" + httpError.getStatusCode)
      case e: Throwable =>
        tool.log(modelid, "模型训练错误", "-1", prop.getProperty("log"))
        LOG.error("模型训练错误\n" + e.printStackTrace())
    }
  }
}
