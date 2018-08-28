package com.zzy

import com.zzy.rf.LOG
import net.sf.json.JSONObject
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

import scala.util.parsing.json.JSON

protected object fit {
    /**
      * 根据正例POS、未标注UNL集合训练模型
      *
      * @param modelid String
      * @param POS     RDD[LabeledPoint]
      * @param UNL     RDD[LabeledPoint]
      * @return (Double,org.apache.spark.mllib.tree.model.RandomForestModel) tmpParam and Model
      */
    def fit(modelid: String, POS: RDD[LabeledPoint], UNL: RDD[LabeledPoint], algo_args: String): (Double, RandomForestModel) = {
        LOG.info("构建模型中")
        tool.log("构建模型中...", "1")
        var c = Double.NaN
        var model_hold_out: RandomForestModel = null
        do {
            val hold_out_ratio = JSONObject.fromObject(algo_args).getString("holdOutRatio").toDouble
            val splits = POS.randomSplit(Array(hold_out_ratio, 1.0 - hold_out_ratio))
            val (p_test, p_train) = (splits(0), splits(1))
            LOG.info(s"train instances ${p_train.count()},test instances ${p_test.count()}")
            // Train a RandomForest model.
            val trainData = p_train.union(UNL)
            if (JSONObject.fromObject(algo_args).getInt("categoricalFeaturesInfo") == 1) {
                LOG.info("categoricalFeaturesInfo=1")
                model_hold_out = RandomForest.trainClassifier(trainData, JSONObject.fromObject(algo_args).getInt("numClasses"), Map[Int, Int](), JSONObject.fromObject(algo_args).getInt("numTrees"), JSONObject.fromObject(algo_args).getString("featureSubsetStrategy"), JSONObject.fromObject(algo_args).getString("impurity"), JSONObject.fromObject(algo_args).getInt("maxDepth"), JSONObject.fromObject(algo_args).getInt("maxBins"))
                val hold_out_predictions = tool.predict(p_test, model_hold_out)
                c = hold_out_predictions.sum() / hold_out_predictions.count()
            }
            else {
                val map = JSON.parseFull(JSONObject.fromObject(JSONObject.fromObject(algo_args).get("categoricalFeaturesInfo")).toString()).get.asInstanceOf[Map[Int, Int]]
                LOG.info(s"categoricalFeaturesInfo ${map.toList}")
                model_hold_out = RandomForest.trainClassifier(trainData, JSONObject.fromObject(algo_args).getInt("numClasses"), map, JSONObject.fromObject(algo_args).getInt("numTrees"), JSONObject.fromObject(algo_args).getString("featureSubsetStrategy"), JSONObject.fromObject(algo_args).getString("impurity"), JSONObject.fromObject(algo_args).getInt("maxDepth"), Math.max(JSONObject.fromObject(algo_args).getInt("maxBins"), map.values.max))
                val hold_out_predictions = tool.predict(p_test, model_hold_out)
                c = hold_out_predictions.sum() / hold_out_predictions.count()
            }
            LOG.info("c is " + c)
            if (c.isNaN) {
                LOG.error("C is Nan")
            }
        } while (c.isNaN)
        tool.log("模型构建完成", "1")
        (c, model_hold_out)
    }
}
