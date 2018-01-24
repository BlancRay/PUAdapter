import java.io.FileReader

import com.zzy.tagModel.{LOG, prop}
import net.sf.json.{JSONArray, JSONObject}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.io.Source

object main {

  def main(args: Array[String]): Unit = {
    println("调用算法训练模型")
    val modelid = "1"
    //设置hadoop目录
    System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
    val sc = new SparkContext(new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]"))
    val p = readData(sc, sc.textFile("E:\\xulei\\zhiziyun\\model\\test\\train_P_label").collect(), "P")
    val u = readData(sc, sc.textFile("E:\\xulei\\zhiziyun\\model\\test\\train_U").collect(), "U")

    val categoryInfos = Source.fromFile("E:\\xulei\\workspace\\ZZY\\attrinfo.txt").getLines.next().replace("{", "").replace("}", "").replace(" ", "").split(",")
    val categoryInfo = mutable.Map[Int, Int]()
    categoryInfos.foreach { each =>
      val tmp = each.split("=")
      categoryInfo.put(tmp(0).toInt, tmp(1).toInt)
    }


    //    //train model use p and u,return arg c and model
    val (estC, model) = fit(p, u, categoryInfo.toMap)
    //
    //    LOG.info("模型训练完成，正在保存模型")
    //
    //    LOG.info("开始计算模型影响因子")
    val feature_importance = importance.importance(model.trees, 2000)
    //
    //    //setInfluenceFacByModelId
    val importance_mapArray = new JSONArray()
    feature_importance.keys.foreach { i =>
      val json_obj = new JSONObject()
      json_obj.put("tagIndex", i.toString)
      json_obj.put("instensity", feature_importance(i).toString)
      importance_mapArray.add(json_obj)
    }
    //
    println(importance_mapArray.toString)
    //    LOG.info("写入模型影响因子")
    //    tool.postArrayToURL(modelid, prop.getProperty("influence"), importance_mapArray)
    //
    //    LOG.info("开始计算相似度")
    val proba = model_test.evaluate(model, estC, modelid, sc, 2000).collect()
    //    tool.log(modelid, "生成相似度与个体数量关系", "1", prop.getProperty("log"))
    val similarity = similiarity.statistics(proba, 0.001, 0.1).toMap
    //
    //    //callBackSimilar
    //    // todo 数据量较大，需要分批写入
    val similarity_mapArray = new JSONArray()
    similarity.keys.foreach { i =>
      val json_obj = new JSONObject()
      json_obj.put("similar", i.toString)
      json_obj.put("num", similarity(i).toString)
      similarity_mapArray.add(json_obj)
    }
    println(similarity_mapArray.toString)
  }

  //
  def readData(sc: SparkContext, data: Array[String], dtype: String) = {
    val lp = new Array[LabeledPoint](data.length)
    var i = 0
    data.foreach { eachdata =>
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


  def fit(POS: RDD[LabeledPoint], UNL: RDD[LabeledPoint], categoryInfo: Map[Int, Int]) = {
    val hold_out_ratio = 0.2
    var c = Double.NaN
    var model_hold_out: RandomForestModel = null
    val splits = POS.randomSplit(Array(hold_out_ratio, 1.0 - hold_out_ratio))
    val (p_test, p_train) = (splits(0), splits(1))
    // Train a RandomForest model.
    val trainData = p_train.union(UNL)

    model_hold_out = RandomForest.trainClassifier(trainData, 2, categoryInfo, 10, "auto", "gini", 15, 100)
    val hold_out_predictions = tool.predict(p_test, model_hold_out)
    c = hold_out_predictions.sum() / hold_out_predictions.count()
    if (c.isNaN) {
      println("C is Nan")
    }
    (c, model_hold_out)
  }
}