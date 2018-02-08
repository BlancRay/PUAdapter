package com.zzy

import java.util
import java.util.Properties

import net.sf.json.{JSONArray, JSONObject}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object ModelPredict_test {
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
    val estC = JSONObject.fromObject(modelinfo).get("tmp_param").toString.toDouble

    //    val (_, data) = tool.read_convert(modelid, "N_SOURCE", sc,TagArray.size())
    //
    //    sys.exit()

    val feature = new Array[Double](TagArray.size())
    println(feature.length)
    //    33,34;157,158;53,54;7,8
    //    137,138;81,82;
    //    feature(80)=100.0
    //    feature(81)=1.0
    feature(136) = 240
    feature(137) = 0.5
    val lp = new Array[LabeledPoint](1)
    lp.update(0, LabeledPoint(0, Vectors.dense(feature)))
    val test_data = sc.makeRDD(lp)
    val prediction = predict(test_data, model)
    val proba = prediction.map(_ / estC).collect()
    println(proba(0))
    //  test end
  }

  /**
    * 计算模型对数据的预测概率
    *
    * @param points RDD[LabeledPoint]
    * @param model  RandomForestModel
    * @return RDD[Double] 预测为正例的概率
    */
  def predict(points: RDD[LabeledPoint], model: RandomForestModel): RDD[Double] = {
    val numTrees = model.trees.length
    val trees = points.sparkContext.broadcast(model.trees)
    points.map { point =>
      trees.value.map(_.predict(point.features)).sum / numTrees
    }
  }
}
