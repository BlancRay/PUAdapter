
package com.zzy

import java.util
import java.util.Properties

import net.sf.json.{JSONArray, JSONObject}
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.{ConnectionFactory, Scan}
import org.apache.hadoop.hbase.filter.PrefixFilter
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.protobuf.ProtobufUtil
import org.apache.hadoop.hbase.util.{Base64, Bytes}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object read_convert_test {
  val prop = new Properties()
  val modelIdMap = new util.HashMap[String, String]()

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
    val modelid = "16"
    val sc = new SparkContext(new SparkConf().setAppName("importance_test").setMaster("local[4]"))
    val path = this.getClass.getResourceAsStream("/model.properties")
    prop.load(path)
    //read model info
    //    val params = new util.HashMap[String, String]()
    modelIdMap.put("key4token", "dmp")
    modelIdMap.put("modelId", modelid)
    val modelinfo = JSONObject.fromObject(tool.postDataToURL(prop.getProperty("model_info"), modelIdMap)).get("outBean").toString
    val model = RandomForestModel.load(sc, prop.getProperty("hdfs_dir") + JSONObject.fromObject(modelinfo).get("model_dir") + "/" + modelid)
    //  test beginning
    val TagIndexInfo = tool.postDataToURL(prop.getProperty("tagindex"), modelIdMap)
    val TagArray = JSONArray.fromObject(JSONObject.fromObject(TagIndexInfo).get("result"))
    val estC = JSONObject.fromObject(modelinfo).get("tmp_param").toString.toDouble

    val (_, data) = read_convert(modelid, "N_SOURCE", sc, TagArray.size())

    //  test end
  }

  /**
    *
    * @param Type String 数据来源，P/U/N
    * @param sc   SparkContext
    * @return (RDD[String],RDD[LabeledPoint]) (GID,数据)
    */
  def read_convert(modelid: String, Type: String, sc: SparkContext, nbTagFeatures: Int): (RDD[String], RDD[LabeledPoint]) = {
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
      (Bytes.toString(each.getRow), Bytes.toString(each.getValue("info".getBytes, "feature".getBytes)))
    }.take(count.toInt)
    val gid = new Array[String](count.toInt)
    val lp = new Array[LabeledPoint](count.toInt)
    var i = 0

    val factors = Array.range(0, nbTagFeatures)
    res.foreach { case (rowkey, hbaseresult) =>
      gid(i) = rowkey
      val GID_TAG_split = new scala.collection.mutable.HashMap[Int, Double]()
      if (hbaseresult != "") {
        val GID_TAG_SET = hbaseresult.split(";")
        GID_TAG_SET.foreach { each =>
          val tag_split = each.split(":")
          GID_TAG_split.put(tag_split(0).toInt, tag_split(1).toDouble)
        }
      }
      //      println(rowkey,hbaseresult)
      //从MODEL_TAG_INDEX表中读取modelID的所有标签 getTagIndexInfoByModelId

      //
      val feature = new Array[Double](factors.length)
      factors.foreach { each =>
        if (GID_TAG_split.contains(each + 1)) {
          feature(each) = GID_TAG_split(each + 1)
        } else {
          feature(each) = 0
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
