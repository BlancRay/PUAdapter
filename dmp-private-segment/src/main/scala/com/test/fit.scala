package com.test

import com.test.rf.{LOG, prop}
import net.sf.json.JSONObject
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Algo.Classification
import org.apache.spark.mllib.tree.configuration.QuantileStrategy.Sort
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

import scala.util.Random
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
        val (hold_out_ratio, numClasses, numTrees, featureSubsetStrategy, impurity, maxDepth) = read_args(algo_args)
        val splits = POS.randomSplit(Array(hold_out_ratio, 1.0 - hold_out_ratio))
        val (p_test, p_train) = (splits(0), splits(1))
        LOG.info(s"train instances ${p_train.count()},test instances ${p_test.count()}")
        // Train a RandomForest model.
        var trainData = p_train.union(UNL)
        val numPartition = prop.getProperty("train_partitions").toInt
        if (numPartition == -1)
            trainData = trainData.repartition((trainData.count() / 1000).toInt)
        else
            trainData = trainData.repartition(numPartition)
        LOG.info(s"训练数据Partition=${trainData.getNumPartitions}")

        var maxBins: Int = 0
        var map = Map[Int, Int]()
        if (JSONObject.fromObject(algo_args).getInt("categoricalFeaturesInfo") == 1) {
            LOG.info("categoricalFeaturesInfo=1")
            maxBins = JSONObject.fromObject(algo_args).getInt("maxBins")
        } else {
            map = JSON.parseFull(JSONObject.fromObject(JSONObject.fromObject(algo_args).get("categoricalFeaturesInfo")).toString()).get.asInstanceOf[Map[Int, Int]]
            LOG.info(s"categoricalFeaturesInfo ${map.toList}")
            maxBins = Math.max(JSONObject.fromObject(algo_args).getInt("maxBins"), map.values.max)

        }

        val strategy = new Strategy(Classification, impurity, maxDepth, numClasses, maxBins, Sort, Map[Int, Int](), maxMemoryInMB = 512)
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

    def read_args(args: String): (Double, Int, Int, String, Impurity, Int) = {
        val hold_out_ratio = JSONObject.fromObject(args).getString("holdOutRatio").toDouble
        val numClasses: Int = JSONObject.fromObject(args).getInt("numClasses")
        val numTrees: Int = JSONObject.fromObject(args).getInt("numTrees")
        val featureSubsetStrategy: String = JSONObject.fromObject(args).getString("featureSubsetStrategy")
        val impurity = fromString(JSONObject.fromObject(args).getString("impurity"))
        val maxDepth: Int = JSONObject.fromObject(args).getInt("maxDepth")
        (hold_out_ratio, numClasses, numTrees, featureSubsetStrategy, impurity, maxDepth)
    }
}
