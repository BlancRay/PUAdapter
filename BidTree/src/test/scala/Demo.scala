import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Algo.Classification
import org.apache.spark.mllib.tree.configuration.FeatureType.{Categorical, Continuous}
import org.apache.spark.mllib.tree.configuration.QuantileStrategy.Sort
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.model.{Node, Split}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author xulei@zhizhiyun.com
  * @version 1.0
  */
object Demo {
    val pwd = "E:\\xulei\\spark\\"

    def main(args: Array[String]) {
        System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
        val sc = new SparkContext(new SparkConf().setAppName("ProbabilityDecisionTree").setMaster("local[4]"))
        val data = MLUtils.loadLibSVMFile(sc, pwd + "iris.txt", 4)
        val splits = data.randomSplit(Array(0.7, 0.3))
        val (trainingData, testData) = (splits(0), splits(1))

        val maxDepth = 2
        val numClasses = 3
        val maxBins = 40
        val minInstancesPerNode = 4
        val categoricalFeaturesInfo = Map[Int, Int]()
        val strategy = new Strategy(Classification, Gini, maxDepth, numClasses, maxBins, Sort, categoricalFeaturesInfo, minInstancesPerNode)
        val dt = RandomForest.trainClassifier(trainingData, strategy, 1, "all", 0)

        val tree = dt.trees(0)
        println(subtreeToString(tree.topNode)) //print tree

        val prob = testData.map { point =>
            predict(tree.topNode, point.features) //predict test Instances and return Probability
        }
        println(prob.collect().toList.toString())
    }

    /**
      * 遍历树的节点，输出决策规则和概率
      *
      * @param topNode      节点
      * @param indentFactor 缩进
      * @return String if...else...
      */
    def subtreeToString(topNode: Node, indentFactor: Int = 0): String = {
        def splitToString(split: Split, left: Boolean): String = {
            split.featureType match {
                case Continuous => if (left) {
                    s"(feature ${split.feature} <= ${split.threshold})"
                } else {
                    s"(feature ${split.feature} > ${split.threshold})"
                }
                case Categorical => if (left) {
                    s"(feature ${split.feature} in ${split.categories.mkString("{", ",", "}")})"
                } else {
                    s"(feature ${split.feature} not in ${split.categories.mkString("{", ",", "}")})"
                }
            }
        }

        val prefix: String = " " * indentFactor
        if (topNode.isLeaf) {
            if (topNode.predict.predict == 0.0) //if Instance predict Negative
                prefix + "leaf_name:\" " + topNode.id + "\" \n" + prefix + "value: " + (1 - topNode.predict.prob) + "\n"
            else //else Instance is Positive
                prefix + "leaf_name:\" " + topNode.id + "\" \n" + prefix + "value: " + topNode.predict.prob + "\n"
        } else {
            prefix + s"If ${splitToString(topNode.split.get, left = true)}\n" +
                subtreeToString(topNode.leftNode.get, indentFactor + 1) +
                prefix + s"Else ${splitToString(topNode.split.get, left = false)}\n" +
                subtreeToString(topNode.rightNode.get, indentFactor + 1)
        }
    }

    /**
      * 递归查询输入数据的预测概率
      *
      * @param node     节点
      * @param features vector
      * @return probability of label 1
      */
    def predict(node: Node, features: Vector): Double = {
        if (node.isLeaf) {
            if (node.predict.predict == 0)
                1 - node.predict.prob
            else
                node.predict.prob
        } else {
            if (node.split.get.featureType == Continuous) {
                if (features(node.split.get.feature) <= node.split.get.threshold) {
                    predict(node.leftNode.get, features)
                } else {
                    predict(node.rightNode.get, features)
                }
            } else {
                if (node.split.get.categories.contains(features(node.split.get.feature))) {
                    predict(node.leftNode.get, features)
                } else {
                    predict(node.rightNode.get, features)
                }
            }
        }
    }
}
