package offline

import java.io.{File, FileOutputStream, PrintWriter}

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.io.Source

object unit_train {
  val pwd = "E:\\xulei\\zhiziyun\\model\\test\\test7\\"
  def main(args: Array[String]): Unit = {
    println("调用算法训练模型")
    //设置hadoop目录
    System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
    val sc = new SparkContext(new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]"))
    val p = readData(sc, sc.textFile(pwd + "train_P_label").collect(), "P")
    val u = readData(sc, sc.textFile(pwd + "train_U").collect(), "U")

    val categoryInfos = Source.fromFile(pwd + "attrInfo.txt").getLines
    val categoryInfo = mutable.Map[Int, Int]()
    categoryInfos.foreach { each =>
      val tmp = each.split("\t")
      categoryInfo.put(tmp(0).toInt, tmp(1).toInt)
    }

    //    //train model use p and u,return arg c and model
    val (estC, model) = fit(p, u, categoryInfo.toMap)
    //
    //    LOG.info("模型训练完成，正在保存模型")
    val saveC = new PrintWriter(new FileOutputStream(pwd + "estC"))
    saveC.write(estC.toString)
    saveC.close()
    val modelFile = new File(pwd + "model")
    dirDel(modelFile)
    model.save(sc, pwd + "model")

  }

  def dirDel(path: File) {
    if (!path.exists())
      return
    else if (path.isFile) {
      path.delete()
      println(path + ":  文件被删除")
      return
    }
    val file: Array[File] = path.listFiles()
    for (d <- file) {
      dirDel(d)
    }
    path.delete()
    println(path + ":  目录被删除")
  }

  def readData(sc: SparkContext, data: Array[String], dtype: String): RDD[LabeledPoint] = {
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


  def fit(POS: RDD[LabeledPoint], UNL: RDD[LabeledPoint], categoryInfo: Map[Int, Int]): (Double, RandomForestModel) = {
    val hold_out_ratio = 0.2
    var c = Double.NaN
    var model_hold_out: RandomForestModel = null
    val splits = POS.randomSplit(Array(hold_out_ratio, 1.0 - hold_out_ratio))
    val (p_test, p_train) = (splits(0), splits(1))
    // Train a RandomForest model.
    val trainData = p_train.union(UNL)

    model_hold_out = RandomForest.trainClassifier(trainData, 2, categoryInfo, 50, "auto", "gini", 25, 200)
    val hold_out_predictions = tool.predict(p_test, model_hold_out)
    c = hold_out_predictions.sum() / hold_out_predictions.count()
    if (c.isNaN) {
      println("C is Nan")
    }
    (c, model_hold_out)
  }
}
