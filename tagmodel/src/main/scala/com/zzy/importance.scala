package com.zzy

import com.zzy.tagModel.LOG
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, Node}

import scala.collection.mutable

protected object importance {
  /**
    * 计算特征重要性
    *
    * @param trees Array[DecisionTreeModel] 随机森林子树
    * @return List[(Int, Double)] 特征重要性[(特征序号，重要性)]
    */
  def importance(trees: Array[DecisionTreeModel], nbTagFeatures: Int) = {
    val f_i = new scala.collection.mutable.HashMap[Int, Double]()
    val feature_importance = mutable.Map[Int, Double]()
    for (i <- 1 to nbTagFeatures) {
      feature_importance.put(i, 0.0)
    }
    for (tree <- trees) {
      if (!tree.topNode.isLeaf) {
        //        println("#########################")
        //        println(tree.topNode.id,tree.topNode.split.get.feature)
        val f_i_tree = scan(tree.topNode, 1.0)
        f_i_tree.foreach { e =>
          if (f_i.contains(e._1)) {
            val curImpurity: Double = f_i(e._1) + e._2
            f_i.put(e._1, curImpurity)
          }
          else
            f_i.put(e._1, e._2)
        }
      }
    }
    f_i.foreach { i =>
      if (i._2.isNaN || i._2.isInfinity) //debug
        LOG.error("f_i.2 is " + i._2)
      f_i.put(i._1, i._2 / trees.length)
    }

    var f_i_sum: Double = 0.0
    f_i.foreach { each =>
      //      println(f_i_sum.isNaN)
      if (each._2.isNaN || each._2.isInfinity) //debug
        LOG.error("error: " + each._2)
      f_i_sum = f_i_sum + each._2
    }

    //    println("f_i_sum is Nan? " + f_i_sum)

    f_i.foreach { e =>
      if (f_i_sum == 0.0 || e._2.isInfinity)
        LOG.error("f_i_sum is 0, will be Nan " + f_i_sum + "," + e._2)
      feature_importance.update(e._1 + 1, e._2 / f_i_sum)
    }
    //    println(feature_importance.head.toString()+",sum "+f_i_sum)
    //    feature_importance.toList.sortBy(_._2)
    feature_importance.toMap
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
    val delta_i = if ((percent * node.stats.get.gain) > 1E-6) percent * node.stats.get.gain else 0.0 //if gain is too small,delta will be 0.0
    if (delta_i.isInfinite || delta_i.isNaN || delta_i == 0.0) //debug
      LOG.error("delta_i is " + delta_i + "\tpercent=" + percent + "\tgain=" + node.stats.get.gain)
    val feature_name = node.split.get.feature
    impurity.put(feature_name, delta_i)
    val left = node.leftNode.get
    val right = node.rightNode.get

    if (!left.isLeaf) {
      var p1 = (node.impurity - node.stats.get.gain - right.impurity) / (left.impurity - right.impurity)
      if (p1.isInfinite) {
        LOG.error("p1 is " + p1 + "\tpercent is " + percent + " p1 isLeaf=" + left.isLeaf + " Node isLeaf=" + node.isLeaf)
        LOG.error("(" + node.impurity + " - " + node.stats.get.gain + " - " + right.impurity + ") / (" + left.impurity + " - " + right.impurity + ")")
      } //debug
      if (p1.isNaN)
        p1 = 0.5
      //in case p1 is Nan
      val left_Map = scan(left, p1 * percent)
      left_Map.foreach { e =>
        if (impurity.contains(e._1)) {
          val curImpurity: Double = impurity(e._1) + e._2
          impurity.put(e._1, curImpurity)
        }
        else
          impurity.put(e._1, e._2)
      }
    } else if (!right.isLeaf) {
      var p2 = (node.impurity - node.stats.get.gain - left.impurity) / (right.impurity - left.impurity)
      if (p2.isInfinite)
        LOG.error("p1 is " + p2 + "\tpercent is " + percent) //debug
      if (p2.isNaN)
        p2 = 0.5
      //in case p2 is Nan
      val right_Map = scan(right, p2 * percent)
      right_Map.foreach { e =>
        if (impurity.contains(e._1)) {
          val curImpurity: Double = impurity(e._1) + e._2
          impurity.put(e._1, curImpurity)
        }
        else
          impurity.put(e._1, e._2)
      }
    }
    impurity
  }
}
