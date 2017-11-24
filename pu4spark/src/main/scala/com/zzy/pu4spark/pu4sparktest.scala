package com.zzy.pu4spark

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext


/**
 * Hello world!
 *
 */
object pu4sparktest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]")//本地测试，上线时修改
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val inputLabelName = "label"
    val srcFeaturesName = "features"
    val outputLabel = "outputLabel"

    val puLearnerConfig = TraditionalPULearnerConfig(0.49, 1000, RandomForestConfig(100))
    val puLearner = puLearnerConfig.build()

    // Load and parse the data file.
    println("load file")
    val (p_gid,p) = readfile("C:/Users/xulei/workspace/as/p",sc)//正例数据
    val (u_gid,u) = readfile("C:/Users/xulei/workspace/as/u",sc)//未标注数据
    val trainData = p.union(u)
    val sqlContext = new SQLContext(sc)
    val df =sqlContext.createDataFrame(trainData) //needed df that contains at least the following columns:
    // binary label for positive and unlabel (inputLabelName)
    // and features assembled as vector (featuresName)
    df.show()
    df.selectExpr("sum(label) as sum").show()
    val weightedDF = puLearner.weight(df, inputLabelName, srcFeaturesName, outputLabel)
      .selectExpr("case curLabel when -1 then 0.0 else curLabel end as curLabel","features")
    //      .drop(outputLabel).drop("indexedFeatures")//.drop("prevLabel")
//    weightedDF.select("curLabel").dropDuplicates().show()
//    weightedDF.selectExpr("sum(curLabel) as sum").show()
//    weightedDF.select("prevLabel","curLabel").dropDuplicates().show()
    val indexer = new StringIndexer()
      .setInputCol("curLabel")
      .setOutputCol("label")
      .fit(weightedDF)
    val transformeddf = indexer.transform(weightedDF)
//    transformeddf.select("label").dropDuplicates().show()
//    transformeddf.selectExpr("sum(label) as sum").show()
    val classifier = new RandomForestClassifier().setMaxBins(32).setMaxDepth(30).setFeatureSubsetStrategy("auto").setImpurity("gini").setNumTrees(512)
    val model = classifier.setLabelCol("label").fit(transformeddf)


    val feature_importance = getFeatureImportance(model)
//    feature_importance.takeRight(10).foreach(each=>println(each))

    val (n_gid,n) = readfile("C:/Users/xulei/workspace/as/n",sc)//测试数据
    val df_n =sqlContext.createDataFrame(n)
    val pred = model.transform(df_n)
//    pred.select("prediction").dropDuplicates().show()



//    val assembler = new VectorAssembler()
//      .setInputCols(df.columns.filter(c => c != "features")) //keep here only feature columns
//      .setOutputCol("featuresName")
//    val pipeline = new Pipeline().setStages(Array(assembler))

//    val preparedDf = pipeline.fit(df).transform(sqlContext.createDataFrame(n))
//    preparedDf.sample(false,0.01).show()
  }

  private def readfile(dir: String,sc:SparkContext) = {
    val n = sc.textFile(dir).collect()
    val gid = new Array[String](n.length)
    val lp = new Array[LabeledPoint](n.length)
    for (i <- n.indices) {
      val split = n(i).split(",")
      gid(i) = split(0)
      val feature = split(2).split(" ")
      val array = new Array[Double](feature.length)
      for (j <- array.indices) {
        array(j) = feature(j).toDouble
      }
      val dv: Vector = Vectors.dense(array)
      lp(i) = LabeledPoint(split(1).toDouble, dv)
    }
    val lpRDD = sc.makeRDD(lp)
    val gidRDD = sc.makeRDD(gid)
    (gidRDD, lpRDD)
  }

  private def getFeatureImportance(model:RandomForestClassificationModel)={
    var i = 0
    val feature_importance = new scala.collection.mutable.HashMap[Int, Double]()
    model.featureImportances.toArray.foreach(f => {
      if (f != 0.0) {
        feature_importance.put(i,f)
      }
      i = i + 1
    }
    )
    feature_importance.toList.sortBy(_._2)
  }
}
