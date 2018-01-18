package com.zzy

import java.util
import java.util.Properties

import net.sf.json.{JSONArray, JSONObject}
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.{SparkConf, SparkContext}

object importance_test {
  val prop = new Properties()
  val modelIdMap = new util.HashMap[String, String]()

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
    val modelid = "16"
    val sc = new SparkContext(new SparkConf().setAppName("importance_test").setMaster("local[4]"))
    val path = this.getClass.getResourceAsStream("/model.properties")
    prop.load(path)
    //read model info
    //    val params = new util.HashMap[String, String]()
    modelIdMap.put("key4token", "dmp")
    modelIdMap.put("modelId", modelid)
    val modelinfo = JSONObject.fromObject(tool.postDataToURL(prop.getProperty("model_info"), modelIdMap)).get("outBean").toString
    val model = RandomForestModel.load(sc, prop.getProperty("hdfs_dir") + JSONObject.fromObject(modelinfo).get("model_dir") + "/" + modelid)
    //  test beginning
    val TagIndexInfo = tool.postDataToURL(prop.getProperty("tagindex"), modelIdMap)
    val TagArray = JSONArray.fromObject(JSONObject.fromObject(TagIndexInfo).get("result"))
    val feature_importance = importance.importance(model.trees, TagArray.size())
    println(feature_importance.toList.sortBy(_._2).toString)
    //  test end
  }

}
