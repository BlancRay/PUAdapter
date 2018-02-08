package offline

import java.io.{FileOutputStream, PrintWriter}

import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

object unit_test {
  val pwd = "E:\\xulei\\zhiziyun\\model\\test\\test7\\"
  val mode = "test" // train or test
  def main(args: Array[String]): Unit = {
    println("开始模型测试")
    //设置hadoop目录
    System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
    val sc = new SparkContext(new SparkConf().setAppName("RandomForestClassificationTest").setMaster("local[4]"))
    println("加载模型及参数")
    val model = RandomForestModel.load(sc, pwd + "model")
    val estC = Source.fromFile(pwd + "estC").getLines().next().toDouble
    println(estC)
    var sb = new StringBuffer()
    //    LOG.info("开始计算模型影响因子")
    val feature_importance = importance.importance(model.trees, 2000)

    //    val feature_importance = importance.featureImportances(model.trees, 2000)    //second method to get feature importance
    sb.append("tagIndex").append(",").append("intensity").append("\n")
    feature_importance.keys.foreach { i =>
      sb.append(i.toString)
        .append(",")
        .append(feature_importance(i).toString)
        .append("\n")
    }
    write2file(sb, pwd + "importance.csv")

    //    LOG.info("写入模型影响因子")
    //    tool.postArrayToURL(modelid, prop.getProperty("influence"), importance_mapArray)
    //
    //    LOG.info("开始计算相似度")
    val proba = model_test.evaluate(model, estC, sc, 2000, pwd + mode + "_All").collect()
    sb = new StringBuffer()
    //    sb.append("gid,prob\n")

    var nbpos = 0
    if (mode == "train")
      nbpos = 700
    else nbpos = 300
    for (i <- proba.indices) {
      if (i < nbpos) // train=700 test=300
        sb.append(1).append(",").append(proba(i)).append("\n")
      else
        sb.append(0).append(",").append(proba(i)).append("\n")
    }
    write2file(sb, pwd + "proba_" + mode + "_all.csv")

    //    tool.log(modelid, "生成相似度与个体数量关系", "1", prop.getProperty("log"))
    val similar = similarity.statistics(proba, 0.001, 1).toMap
    //
    //    //callBackSimilar
    sb = new StringBuffer()
    sb.append("similar,num\n")
    similar.keys.foreach { i =>
      sb.append(i.toString)
        .append(",")
        .append(similar(i).toString)
        .append("\n")
    }
    write2file(sb, pwd + "similarity_" + mode + "_all.csv")
    sb = null
  }

  def write2file(string: StringBuffer, dir: String): Unit = {
    val writer = new PrintWriter(new FileOutputStream(dir))
    writer.print(string)
    writer.close()
  }
}
