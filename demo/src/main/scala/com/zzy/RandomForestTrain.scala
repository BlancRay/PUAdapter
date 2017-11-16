package com.zzy

import java.io._
import java.util.Properties

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, Node, RandomForestModel}

object RandomForestTrain {
  val prop: Properties = new Properties()

  /**
    * 根据参数model_id读取数据、训练模型、输出特征重要性、保存训练好的模型及参数
    *
    * @param args Array[String] 模型id
    */
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]")
    //本地测试，上线时修改
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val arg = "C:/Users/xulei/workspace/as/rf"
    println("read properties")
    val in: InputStream = new BufferedInputStream(new FileInputStream(arg + ".properties"))
    prop.load(in)

    println("start")
    // Load and parse the data file.
    println("load file")
    val (p_gid, p) = new tools().readData(prop.getProperty("p_dir"), sc)
    //正例数据
    val (u_gid, u) = new tools().readData(prop.getProperty("u_dir"), sc) //未标注数据

    println("training model")
    // Train a RandomForest model.
    val (estC, model) = fit(p, u)

    println("Model save")
    save(model, estC, sc)

    sc.stop()
  }

  /**
    * 根据正例POS、未标注UNL集合训练模型
    *
    * @param POS RDD[LabeledPoint]
    * @param UNL RDD[LabeledPoint]
    * @return org.apache.spark.mllib.tree.model.RandomForestModel
    */
  private def fit(POS: RDD[LabeledPoint], UNL: RDD[LabeledPoint]) = {
    val splits = POS.randomSplit(Array(0.5, 0.5))
    val (p_test, p_train) = (splits(0), splits(1))
    // Train a RandomForest model.
    val trainData = p_train.union(UNL)
    val model_hold_out = RandomForest.trainClassifier(trainData, 2, Map[Int, Int](), 512, "auto", "gini", 30, 32)

    val feature_importance = importance(model_hold_out.trees)
    new tools().List2file(feature_importance, prop.getProperty("model_importance_dir"))

    val hold_out_predictions = new tools().predict(p_test, model_hold_out)
    val c = hold_out_predictions.sum() / hold_out_predictions.count()
    println("c is " + c)
    (c, model_hold_out)
  }

  /**
    * 计算特征重要性
    *
    * @param trees Array[DecisionTreeModel] 随机森林子树
    * @return List[(Int, Double)] 特征重要性[(特征序号，重要性)]
    */
  private def importance(trees: Array[DecisionTreeModel]) = {
    val f_i = new scala.collection.mutable.HashMap[Int, Double]()
    val feature_importance = new scala.collection.mutable.HashMap[Int, Double]()
    for (tree <- trees) {
      val f_i_tree = scan(tree.topNode, 1.0)
      f_i_tree.foreach(e => {
        if (f_i.contains(e._1)) {
          val curImpurity: Double = f_i(e._1) + e._2
          f_i.put(e._1, curImpurity)
        }
        else
          f_i.put(e._1, e._2)
      })
    }
    f_i.foreach(i => {
      if (i._2.isNaN)
        sys.error("f_i.2 is Nan")
      f_i.put(i._1, i._2 / trees.length)
    })

    var f_i_sum: Double = 0.0
    f_i.foreach(each => {
      //      println(f_i_sum.isNaN)
      if (each._2.isNaN || f_i_sum.isNaN)
        sys.error("error: " + each._2 + f_i_sum)
      f_i_sum = f_i_sum + each._2
    })

    //    println("f_i_sum is Nan? " + f_i_sum)

    f_i.foreach(e => {
      feature_importance.put(e._1, e._2 / f_i_sum)
    })
    //    println(feature_importance.head.toString()+",sum "+f_i_sum)
    feature_importance.toList.sortBy(_._2)
  }

  /**
    * 计算当前节点下的特征重要性
    *
    * @param node    Node 节点
    * @param percent Double 数据到达该节点的概率
    * @return mutable.HashMap[Int, Double] 特征重要性[特征序号，重要性]
    */
  private def scan(node: Node, percent: Double): scala.collection.mutable.HashMap[Int, Double] = {
    val impurity = new scala.collection.mutable.HashMap[Int, Double]()
    val left = node.leftNode.get
    val right = node.rightNode.get
    var p1 = (node.impurity - node.stats.get.gain - right.impurity) / (left.impurity - right.impurity)
    var p2 = (node.impurity - node.stats.get.gain - left.impurity) / (right.impurity - left.impurity)
    val delta_i = percent * node.stats.get.gain
    val feature_name = node.split.get.feature
    impurity.put(feature_name, delta_i)

    if (!left.isLeaf) {
      if (p1.isNaN)
        p1 = 0.5
      //in case p1 is Nan
      //        sys.error("p1 is Nan" + left.impurity + right.impurity)
      val left_Map = scan(left, p1 * percent)
      left_Map.foreach(e => {
        if (impurity.contains(e._1)) {
          val curImpurity: Double = impurity(e._1) + e._2
          impurity.put(e._1, curImpurity)
        }
        else
          impurity.put(e._1, e._2)
      })
    } else if (!right.isLeaf) {
      if (p2.isNaN)
        p2 = 0.5
      //in case p2 is Nan
      //        sys.error("p2 is Nan" + left.impurity + right.impurity)
      val right_Map = scan(right, p2 * percent)
      right_Map.foreach(e => {
        if (impurity.contains(e._1)) {
          val curImpurity: Double = impurity(e._1) + e._2
          impurity.put(e._1, curImpurity)
        }
        else
          impurity.put(e._1, e._2)
      })
    }
    impurity
  }

  /**
    * 保存模型文件和参数c
    *
    * @param model org.apache.spark.mllib.tree.model.RandomForestModel模型
    * @param estC  Double 参数c
    * @param sc    SparkContext
    */
  private def save(model: RandomForestModel, estC: Double, sc: SparkContext) {
    //    model.save(sc, prop.getProperty("model_dir")) //模型保存在hdfs上
    val writer = new PrintWriter(new File(prop.getProperty("model_c"))) //模型参数c
    writer.println(estC)
    writer.close()
  }
}
