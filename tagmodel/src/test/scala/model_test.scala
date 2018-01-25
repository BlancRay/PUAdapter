import main.readData
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

protected object model_test {
  /**
    * 根据参数获取模型参数，读取测试数据、模型及参数c、输出预测概率
    *
    * @param model RandomForestModel 模型
    * @param estC  Double c
    * @return RDD[Double] 预测概率
    */
  def evaluate(model: RandomForestModel, estC: Double, sc: SparkContext, nbTagFeatures: Int): RDD[Double] = {
    //    tool.log(modelid, "生成模型测试数据", "1", prop.getProperty("log"))
    //    val (_, test_data) = tool.read_convert(modelid, "N_SOURCE", sc, nbTagFeatures)
    val test_data = readData(sc, sc.textFile("E:\\xulei\\zhiziyun\\model\\test\\train_P_label").collect(), "N")
    println("N_SOURCE数量:" + test_data.count().toString)
    //    tool.log(modelid, "模型测试中", "1", prop.getProperty("log"))
    val prediction = predict(test_data, model)
    val proba = prediction.map(_ / estC)
    //    tool.log(modelid, "模型测试完成", "1", prop.getProperty("log"))
    proba
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
