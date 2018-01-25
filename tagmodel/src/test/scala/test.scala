import java.io.{FileOutputStream, PrintWriter}

import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

object test {
  def main(args: Array[String]): Unit = {
    println("开始模型测试")
    //设置hadoop目录
    System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
    val sc = new SparkContext(new SparkConf().setAppName("RandomForestClassificationTest").setMaster("local[4]"))
    println("加载模型及参数")
    val model = RandomForestModel.load(sc, "E:\\xulei\\zhiziyun\\model\\test\\model")
    val estC = Source.fromFile("E:\\xulei\\zhiziyun\\model\\test\\estC").getLines().next().toDouble
    println(estC)
    var sb = new StringBuffer()
    //    LOG.info("开始计算模型影响因子")
    val feature_importance = importance.importance(model.trees, 2000)
    sb.append("tagIndex").append(",").append("instensity").append("\n")
    feature_importance.keys.foreach { i =>
      sb.append(i.toString)
        .append(",")
        .append(feature_importance(i).toString)
        .append("\n")
    }
    write2file(sb, "E:\\xulei\\zhiziyun\\model\\test\\importance.csv")
    //    LOG.info("写入模型影响因子")
    //    tool.postArrayToURL(modelid, prop.getProperty("influence"), importance_mapArray)
    //
    //    LOG.info("开始计算相似度")
    val proba = model_test.evaluate(model, estC, sc, 2000).collect()
    sb = new StringBuffer()
    sb.append("gid,prob\n")
    for (i <- proba.indices) {
      sb.append(i).append(",").append(proba(i)).append("\n")
    }
    write2file(sb, "E:\\xulei\\zhiziyun\\model\\test\\proba.csv")

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
    write2file(sb, "E:\\xulei\\zhiziyun\\model\\test\\similarity.csv")
    sb = null
  }

  def write2file(string: StringBuffer, dir: String): Unit = {
    val writer = new PrintWriter(new FileOutputStream(dir))
    writer.print(string)
    writer.close()
  }
}
