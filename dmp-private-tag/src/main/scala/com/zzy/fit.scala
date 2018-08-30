package com.zzy

import com.zzy.tagModel.LOG
import net.sf.json.JSONObject
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Algo.Classification
import org.apache.spark.mllib.tree.configuration.QuantileStrategy.Sort
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity, Variance}
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

import scala.util.Random

protected object fit {
    /**
      * 根据正例POS、未标注UNL集合训练模型
      *
      * @param POS RDD[LabeledPoint]
      * @param UNL RDD[LabeledPoint]
      * @return org.apache.spark.mllib.tree.model.RandomForestModel
      */
    def fit(POS: RDD[LabeledPoint], UNL: RDD[LabeledPoint], algo_args: String, categoryInfo: Map[Int, Int]): (Double, RandomForestModel) = {
        LOG.info("构建模型中")
        tool.log("构建模型中...", "1")
        val (hold_out_ratio, numClasses, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins) = read_args(algo_args, categoryInfo)
        var c = Double.NaN
        var model_hold_out: RandomForestModel = null
        val splits = POS.randomSplit(Array(hold_out_ratio, 1.0 - hold_out_ratio))
        val (p_test, p_train) = (splits(0), splits(1))
        val trainData = p_train.union(UNL)
        LOG.info(s"train instances ${p_train.count()},test instances ${p_test.count()}")

        //        Train a RandomForest model.
        val strategy = new Strategy(Classification, impurity, maxDepth, numClasses, maxBins, Sort, categoryInfo, maxMemoryInMB = 512)
        do {
            model_hold_out = RandomForest.trainClassifier(trainData, strategy, numTrees, featureSubsetStrategy, Random.nextInt())
            val hold_out_predictions = tool.predict(p_test, model_hold_out)
            c = hold_out_predictions.sum() / hold_out_predictions.count()
            LOG.info("c is " + c)
            if (c.isNaN) {
                LOG.error("C is Nan")
            }
        } while (c.isNaN)
        tool.log("模型构建完成", "1")
        (c, model_hold_out)
    }

    def fromString(name: String): Impurity = name match {
        case "gini" => Gini
        case "entropy" => Entropy
        case "variance" => Variance
        case _ => throw new IllegalArgumentException(s"Did not recognize Impurity name: $name")
    }

    def read_args(args: String, categoryInfo: Map[Int, Int]): (Double, Int, Int, String, Impurity, Int, Int) = {
        val hold_out_ratio = JSONObject.fromObject(args).getString("holdOutRatio").toDouble
        val numClasses: Int = JSONObject.fromObject(args).getInt("numClasses")
        val numTrees: Int = JSONObject.fromObject(args).getInt("numTrees")
        val featureSubsetStrategy: String = JSONObject.fromObject(args).getString("featureSubsetStrategy")
        val impurity = fromString(JSONObject.fromObject(args).getString("impurity"))
        val maxDepth: Int = JSONObject.fromObject(args).getInt("maxDepth")
        val maxBins: Int = Math.max(JSONObject.fromObject(args).getInt("maxBins"), categoryInfo.values.max)
        (hold_out_ratio, numClasses, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    }
}
