package com.zzy

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable


/**
  * Hello world!
  *
  */
object pu4sparktest {
    //    val modelIdMap = new util.HashMap[String, String]()
    //    val prop = new Properties()
    def main(args: Array[String]): Unit = {
        System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
        //        val path = this.getClass.getResourceAsStream("/model.properties")
//        prop.load(path)
        val sparkSession = new SparkSession.Builder().appName("RandomForestClassificationTrain").master("local[4]").getOrCreate() //本地测试，上线时修改
        sparkSession.sparkContext.setLogLevel("ERROR")
        val inputLabelName = "label"
        val srcFeaturesName = "features"
        val outputLabel = "outputLabel"

        val puLearnerConfig = TraditionalPULearnerConfig(0.49, 1000, RandomForestConfig(100))
        val puLearner = puLearnerConfig.build()

        // Load and parse the data file.
//        val dir=this.getClass.getResource("/")
//        println("load file from "+ dir)
        println("read labeled as pos file")
        val p = tools.readData(sparkSession.sparkContext, sparkSession.sparkContext.textFile("E:\\xulei\\zhiziyun\\model\\test\\pu4spark\\train_P_label"), "P")
        //正例数据
        println("read unlabeled file")
        val u = tools.readData(sparkSession.sparkContext, sparkSession.sparkContext.textFile("E:\\xulei\\zhiziyun\\model\\test\\pu4spark\\train_U"), "U")
        //未标注数据
        val trainData = p.union(u)
        val sqlContext = sparkSession.sqlContext
        val df = sqlContext.createDataFrame(trainData) //needed df that contains at least the following columns:
        // binary label for positive and unlabel (inputLabelName)
        // and features assembled as vector (featuresName)
//        df.show()
        df.selectExpr("count(1) as count").show()
        val weightedDF = puLearner.weight(df, inputLabelName, srcFeaturesName, outputLabel)
            .selectExpr("case curLabel when -1 then 0.0 else curLabel end as curLabel", "features")
        //      .drop(outputLabel).drop("indexedFeatures")//.drop("prevLabel")
        //    weightedDF.select("curLabel").dropDuplicates().show()
        //    weightedDF.selectExpr("sum(curLabel) as sum").show()
        //    weightedDF.select("prevLabel","curLabel").dropDuplicates().show()
        val indexer = new StringIndexer().setInputCol("curLabel").setOutputCol("label").fit(weightedDF)
        val transformeddf = indexer.transform(weightedDF)
        //    transformeddf.select("label").dropDuplicates().show()
        //    transformeddf.selectExpr("sum(label) as sum").show()
        val classifier = new RandomForestClassifier().setMaxBins(32).setMaxDepth(30).setFeatureSubsetStrategy("auto").setImpurity("gini").setNumTrees(512)
        val model = classifier.setLabelCol("label").fit(transformeddf)
        println("model has been fitted,saving model")
        model.save("E:\\xulei\\zhiziyun\\model\\test\\pu4spark\\model")

        println("get feature importance")
        val feature_importance = getFeatureImportance(model)
        //    feature_importance.takeRight(10).foreach(each=>println(each))

        println("read test data")
        val n = tools.readData(sparkSession.sparkContext, sparkSession.sparkContext.textFile("E:\\xulei\\zhiziyun\\model\\test\\pu4spark\\train_All"), "N")
        //测试数据
        val df_n = sqlContext.createDataFrame(n)
        println("get predictions from test data")
        val pred = model.transform(df_n)
        //    pred.select("prediction").dropDuplicates().show()
        println("get similarities")
        val similar = statistics(pred,1,0.01).toMap


        //    val assembler = new VectorAssembler()
        //      .setInputCols(df.columns.filter(c => c != "features")) //keep here only feature columns
        //      .setOutputCol("featuresName")
        //    val pipeline = new Pipeline().setStages(Array(assembler))

        //    val preparedDf = pipeline.fit(df).transform(sqlContext.createDataFrame(n))
        //    preparedDf.sample(false,0.01).show()
    }


    private def getFeatureImportance(model: RandomForestClassificationModel): List[(Int, Double)] = {
        var i = 0
        val feature_importance = new scala.collection.mutable.HashMap[Int, Double]()
        model.featureImportances.toArray.foreach {
            f => {
                if (f != 0.0) {
                    feature_importance.put(i, f)
                }
                i = i + 1
            }
        }
        feature_importance.toList.sortBy(_._2)
    }

    def statistics(proba: DataFrame, threshold: Double, ratio: Double): mutable.Map[Double, Int] = {
        val sn: Int = (1 / threshold).toInt + 1
        val similarity = mutable.Map[Double, Int]()
        var tmp: Double = 0.0
        for (i <- 0 until sn) {
            val nbperson = proba.select("prediction").where("prediction>="+tmp).count()
            similarity.put(tmp, (nbperson / ratio).toInt)
            tmp = threshold + tmp
        }
        similarity
    }
}
