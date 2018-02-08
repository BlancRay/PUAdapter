package online

import online.unit_test.prop
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.{ConnectionFactory, Scan}
import org.apache.hadoop.hbase.filter.PrefixFilter
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.protobuf.ProtobufUtil
import org.apache.hadoop.hbase.util.{Base64, Bytes}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

import scala.collection.mutable

protected object model_test {
  /**
    * 根据参数获取模型参数，读取测试数据、模型及参数c、输出预测概率
    *
    * @param model RandomForestModel 模型
    * @param estC  Double c
    * @return (gid,proba) (Array[String],Array[Double])
    */
  def evaluate(model: RandomForestModel, estC: Double, modelid: String, sc: SparkContext, Features: mutable.Map[Int, Int]): RDD[Double] = {
    //    tool.log(modelid, "生成模型测试数据", "1", prop.getProperty("log"))
    //    val (_, test_data) = tool.read_convert(modelid, "N_SOURCE", sc, Features)
    val (_, test_data) = read_convert(modelid, "N_SOURCE", sc, Features)
    //    LOG.info("N_SOURCE数量:" + test_data.count().toString)
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

  def read_convert(modelid: String, Type: String, sc: SparkContext, Features: mutable.Map[Int, Int]): (RDD[String], RDD[LabeledPoint]) = {
    val HBconf = HBaseConfiguration.create()
    HBconf.set("hbase.zookeeper.property.clientPort", prop.getProperty("hbase_clientPort"))
    HBconf.set("hbase.zookeeper.quorum", prop.getProperty("hbase_quorum")) //"kylin-node4,kylin-node3,kylin-node2"
    if (Type.contains("P"))
      HBconf.set(TableInputFormat.INPUT_TABLE, prop.getProperty("p_source_table"))
    else if (Type.contains("U"))
      HBconf.set(TableInputFormat.INPUT_TABLE, prop.getProperty("u_source_table"))
    else if (Type.contains("N"))
      HBconf.set(TableInputFormat.INPUT_TABLE, prop.getProperty("n_source_table"))
    val prefixFilter = new PrefixFilter(Bytes.toBytes(modelid))
    val scan = new Scan()
    scan.setFilter(prefixFilter)
    val proto = ProtobufUtil.toScan(scan)
    HBconf.set(TableInputFormat.SCAN, Base64.encodeBytes(proto.toByteArray))
    //    HBconf.set(TableInputFormat.SCAN, modelid)

    val conn = ConnectionFactory.createConnection(HBconf)
    //根据Type读取对应数据
    val data_source = sc.newAPIHadoopRDD(HBconf, classOf[TableInputFormat], classOf[ImmutableBytesWritable], classOf[org.apache.hadoop.hbase.client.Result])
    val count = data_source.count()
    //    println(count)
    val res = data_source.map { case (_, each) =>
      //从SOURCE表中读取的某个GID的ROWKEY
      //从SOURCE表中读取的某个GID的column标签集合
      (Bytes.toString(each.getRow), Bytes.toString(each.getValue("info".getBytes, "trait".getBytes)))
    }.take(count.toInt)
    val gid = new Array[String](count.toInt)
    val lp = new Array[LabeledPoint](count.toInt)
    var i = 0

    val factors = Array.range(0, Features.size)
    res.foreach {
      case (rowkey, hbaseresult) =>
        gid(i) = rowkey
        val GID_TAG_split = new mutable.HashMap[Int, Double]()
        if (hbaseresult != "") {
          val GID_TAG_SET = hbaseresult.split(";")
          GID_TAG_SET.foreach { each =>
            val tag_split = each.split(":")
            GID_TAG_split.put(tag_split(0).toInt, tag_split(1).toDouble)
          }
        } else {
          throw new Exception(rowkey + " 的特征数为0!")
        }

        //      println(rowkey,hbaseresult)
        //从MODEL_TAG_INDEX表中读取modelID的所有标签 getTagIndexInfoByModelId

        //
        val feature = new Array[Double](factors.length)
        factors.foreach { each =>
          if (GID_TAG_split.contains(each)) {
            //若数据库中下标从1开始,则为contains(each+1),否则为contains(each)
            feature(each) = GID_TAG_split(each) //GID_TAG_split(each+1)或GID_TAG_split(each)
          } else {
            if (Features(each) == 1) //连续变量长度为1
              feature(each) = -1.0
            else //离散变量长度至少为2(包含一个"NULL")
              feature(each) = 0.0
          }
        }

        val dv: Vector = Vectors.dense(feature)
        if (Type.contains("P"))
          lp.update(i, LabeledPoint(1, dv))
        else
          lp.update(i, LabeledPoint(0, dv))
        i += 1
    }
    conn.close()
    (sc.makeRDD(gid), sc.makeRDD(lp))
  }
}
