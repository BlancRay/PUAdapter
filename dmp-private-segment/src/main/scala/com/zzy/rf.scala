package com.zzy

import java.io.{BufferedInputStream, FileInputStream}
import java.util
import java.util.Properties

import net.sf.json.{JSONArray, JSONObject}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
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
        //        System.setProperty("HADOOP_USER_NAME", "hdfs")
        val conf = new Configuration()
        val hdfs = FileSystem.get(conf)
        LOG.info(hdfs.getUri.toString)

        //        val modelid = "34"
        //        val sc = new SparkContext(new SparkConf().setAppName("dmp-private-segment").setMaster("local[4]"))
        //        prop.load(new FileInputStream("E:\\IdeaProjects\\dmp\\dmp-private-segment\\model.properties"))

        val modelid = args(0)
        val sc = new SparkContext(new SparkConf().setAppName("dmp-private-segment"))
        prop.load(new BufferedInputStream(new FileInputStream(args(1))))
        prop.setProperty("hdfs_dir", hdfs.getUri.toString)

        modelIdMap.put("key4token", "dmp")
        modelIdMap.put("modelId", modelid)

        try {
            tool.log("调用算法训练模型开始", "1")
            //read model info
            tool.log("读取模型参数", "1")
            val modelinfo = JSONObject
                .fromObject(
                    tool.postDataToURL(prop.getProperty("model_info"), modelIdMap)
                )
                .get("outBean")
                .toString
            LOG.info("模型信息:" + modelinfo)
            val algo_args = JSONObject.fromObject(modelinfo).get("algorithm_args").toString
            val model_args = JSONObject.fromObject(modelinfo).get("model_args").toString

            val TagIndexInfo = tool.postDataToURL(prop.getProperty("tagindex"), modelIdMap)
            val TagArray = JSONArray.fromObject(JSONObject.fromObject(TagIndexInfo).get("result"))
            //read p u data
            tool.log("读取种子人群", "1")

            val (_, p) = tool.read_convert(modelid, "P", sc, TagArray.size())
            LOG.info("P_SOURCE数量:" + p.count().toString)

            tool.log("读取训练数据", "1")
            val (_, u) = tool.read_convert(modelid, "U", sc, TagArray.size())
            LOG.info("U_SOURCE数量:" + u.count().toString)
            LOG.info("数据读取完成")

            //train model use p and u,return arg c and model
            val (estC, model) = fit.fit(modelid, p, u, algo_args)

            LOG.info("模型训练完成，正在保存模型")
            tool.save(model, estC, modelinfo, modelid, sc)

            tool.log("模型保存完成，开始计算模型影响因子", "1")
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

            LOG.info("写入模型影响因子")
            tool.postArrayToURL(modelid, prop.getProperty("influence"), importance_mapArray)
            tool.log("模型影响因子保存完成", "1")
            LOG.info(feature_importance.toList.sortBy(_._2).toString)

            LOG.info("开始计算相似度")
            val proba = model_test.evaluate(model, estC, modelid, sc, TagArray.size()).collect()
            tool.log("生成相似度与个体数量关系", "1")
            val similar = similarity
                .statistics(
                    proba,
                    JSONObject.fromObject(model_args).getDouble("delta"),
                    JSONObject.fromObject(modelinfo).getDouble("sigma")
                )
                .toMap
            tool.log("相似度与个体数量关系计算完成，正在保存...", "1")
            LOG.info("相似度计算完成，写入数据库中")

            //callBackSimilar
            var similarity_mapArray = new JSONArray()
            similar.keys.foreach { i =>
                val json_obj = new JSONObject()
                json_obj.put("similar", i.toString)
                json_obj.put("num", similar(i).toString)
                similarity_mapArray.add(json_obj)
                if (similarity_mapArray.size() % 5000 == 0) {
                    tool.postArrayToURL(modelid, prop.getProperty("similar"), similarity_mapArray)
                    similarity_mapArray = new JSONArray()
                    Thread.sleep(1000)
                }
            }
            if (similarity_mapArray.size() > 0) {
                tool.postArrayToURL(modelid, prop.getProperty("similar"), similarity_mapArray)
            }
            tool.log("模型训练完成", "2")
            LOG.info("模型训练完成\n\n\n\n")
        } catch {
            case httpError: HttpStatusException =>
                LOG.error("服务器错误，Status code：" + httpError.getStatusCode)
            case t: Throwable =>
                tool.log("模型训练错误", "-1")
                LOG.error("模型训练错误\n" + t.printStackTrace())
            case e: Exception =>
                tool.log("模型训练错误", "-1")
                LOG.error("模型训练错误\n" + e.printStackTrace())
        } finally
            sc.stop()
    }
}
