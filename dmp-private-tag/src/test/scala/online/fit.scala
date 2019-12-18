package online

import com.google.gson.JsonObject
import com.zzy.tagModel.{LOG, prop}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

protected object fit {
    /**
      * 根据正例POS、未标注UNL集合训练模型
      *
      * @param modelid String
      * @param POS     RDD[LabeledPoint]
      * @param UNL     RDD[LabeledPoint]
      * @return org.apache.spark.mllib.tree.model.RandomForestModel
      */
    def fit(modelid: String, POS: RDD[LabeledPoint], UNL: RDD[LabeledPoint], algo_args: JsonObject, categoryInfo: Map[Int, Int]): (Double, RandomForestModel) = {
        var c = Double.NaN
        var model_hold_out: RandomForestModel = null
        do {
            val hold_out_ratio = algo_args.get("holdOutRatio").getAsDouble
            val splits = POS.randomSplit(Array(hold_out_ratio, 1.0 - hold_out_ratio))
            val (p_test, p_train) = (splits(0), splits(1))
            // Train a RandomForest model.
            val trainData = p_train.union(UNL)
            //      val categoryInfo = tool.getCategoryInfo(modelid)
            model_hold_out = RandomForest.trainClassifier(trainData, algo_args.get("numClasses").getAsInt, categoryInfo, algo_args.get("numTrees").getAsInt, algo_args.get("featureSubsetStrategy").getAsString, algo_args.get("impurity").getAsString, algo_args.get("maxDepth").getAsInt, algo_args.get("maxBins").getAsInt)
            val hold_out_predictions = tool.predict(p_test, model_hold_out)
            c = hold_out_predictions.sum() / hold_out_predictions.count()
            LOG.info("c is " + c)
            if (c.isNaN) {
                LOG.error("C is Nan")
            }
        } while (c.isNaN)
        tool.log(modelid, "模型训练完成", "1", prop.getProperty("log"))
        (c, model_hold_out)
    }
}
