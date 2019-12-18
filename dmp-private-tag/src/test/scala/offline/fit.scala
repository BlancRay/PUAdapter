package offline

import com.google.gson.{Gson, JsonObject}
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
      * @return (Double,org.apache.spark.mllib.tree.model.RandomForestModel) tmpParam and Model
      */
    def fit(modelid: String, POS: RDD[LabeledPoint], UNL: RDD[LabeledPoint], algo_args: JsonObject): (Double, RandomForestModel) = {
        var c = Double.NaN
        var model_hold_out: RandomForestModel = null
        val gson = new Gson()
        do {
            val hold_out_ratio = algo_args.get("holdOutRatio").getAsDouble
            val splits = POS.randomSplit(Array(hold_out_ratio, 1.0 - hold_out_ratio))
            val (p_test, p_train) = (splits(0), splits(1))
            // Train a RandomForest model.
            val trainData = p_train.union(UNL)
            if (algo_args.get("categoricalFeaturesInfo").getAsInt == 1) {
                model_hold_out = RandomForest.trainClassifier(trainData, algo_args.get("numClasses").getAsInt, Map[Int, Int](), algo_args.get("numTrees").getAsInt, algo_args.get("featureSubsetStrategy").getAsString, algo_args.get("impurity").getAsString, algo_args.get("maxDepth").getAsInt, algo_args.get("maxBins").getAsInt)
                val hold_out_predictions = tool.predict(p_test, model_hold_out)
                c = hold_out_predictions.sum() / hold_out_predictions.count()
            }
            else {
                val map = gson.fromJson(algo_args.get("categoricalFeaturesInfo"), classOf[Map[Int, Int]])
                model_hold_out = RandomForest.trainClassifier(trainData, algo_args.get("numClasses").getAsInt, map, algo_args.get("numTrees").getAsInt, algo_args.get("featureSubsetStrategy").getAsString, algo_args.get("impurity").getAsString, algo_args.get("maxDepth").getAsInt, algo_args.get("maxBins").getAsInt)
                val hold_out_predictions = tool.predict(p_test, model_hold_out)
                c = hold_out_predictions.sum() / hold_out_predictions.count()
            }
            println("c is " + c)
            if (c.isNaN) {
                println("C is Nan")
            }
        } while (c.isNaN)
        //    tool.log(modelid, "模型训练完成", "1", prop.getProperty("log"))
        (c, model_hold_out)
    }
}
