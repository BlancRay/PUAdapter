package com.test

import java.text.SimpleDateFormat
import java.util
import java.util.Date

import com.test.rf.{LOG, modelIdMap, prop}
import net.sf.json.{JSONArray, JSONObject}
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.{ConnectionFactory, Scan}
import org.apache.hadoop.hbase.filter._
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.protobuf.ProtobufUtil
import org.apache.hadoop.hbase.util.{Base64, Bytes}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.jsoup.Jsoup

import scala.collection.mutable

object tool {
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

    /**
      *
      * @param Type String 数据来源，P/U/N
      * @param sc   SparkContext
      * @return (RDD[String],RDD[LabeledPoint]) (GID,数据)
      */
    def read_convert(modelid: String, Type: String, sc: SparkContext, nbTagFeatures: Int): (RDD[String], RDD[LabeledPoint]) = {
        val HBconf = HBaseConfiguration.create()
        HBconf.set("hbase.zookeeper.property.clientPort", prop.getProperty("hbase_clientPort"))
        HBconf.set("hbase.zookeeper.quorum", prop.getProperty("hbase_quorum"))
        HBconf.set("zookeeper.znode.parent", prop.getProperty("hbase_znode"))
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

        val conn = ConnectionFactory.createConnection(HBconf)
        //根据Type读取对应数据
        val data_source = sc.newAPIHadoopRDD(HBconf, classOf[TableInputFormat], classOf[ImmutableBytesWritable], classOf[org.apache.hadoop.hbase.client.Result])
        val res = data_source.map { case (_, each) =>
            //从SOURCE表中读取的某个GID的ROWKEY
            //从SOURCE表中读取的某个GID的column标签集合
            (Bytes.toString(each.getRow), Bytes.toString(each.getValue("info".getBytes, "feature".getBytes)))
        }

        LOG.info("所有标签特征数量:" + nbTagFeatures.toString)
        val factors = Array.range(0, nbTagFeatures)
        val result = res.map { res_each =>
            val gid = res_each._1
            val hbaseresult = res_each._2
            val GID_TAG_split = new mutable.HashMap[Int, Double]()
            if (hbaseresult != "") {
                val GID_TAG_SET = hbaseresult.split(";")
                var is_time = true
                GID_TAG_SET.foreach { set_each =>
                    val tag_split = set_each.split(":")
                    if (is_time) {
                        GID_TAG_split.put(tag_split(0).toInt, tag_split(1).toDouble / 3600000)
                        is_time = false
                    } else {
                        GID_TAG_split.put(tag_split(0).toInt, tag_split(1).toDouble)
                        is_time = true
                    }
                }
            } else {
                LOG.error(gid + " 的特征数为0!")
                log(gid + " 的特征数为0!", "-1")
                throw new Exception(gid + " 的特征数为0!")
            }

            val feature = new Array[Double](factors.length)
            factors.foreach { f_each =>
                if (GID_TAG_split.contains(f_each + 1)) {
                    feature(f_each) = GID_TAG_split(f_each + 1)
                } else {
                    feature(f_each) = 0
                }
            }

            val dv: Vector = Vectors.dense(feature)
            var lp: LabeledPoint = null
            if (Type.contains("P")) {
                lp = LabeledPoint(1, dv)
            }
            else {
                lp = LabeledPoint(0, dv)
            }
            (gid, lp)
        }
        LOG.info("sampled %s like %s".format(Type, result.take(1).toList.toString()))
        conn.close()
        (result.map(_._1), result.map(_._2))
    }

    /**
      * 保存模型文件和参数c
      *
      * @param model   org.apache.spark.mllib.tree.model.RandomForestModel模型
      * @param estC    Double 参数c
      * @param modelid String 模型id
      */
    def save(model: RandomForestModel, estC: Double, model_info: String, modelid: String, sc: SparkContext) {
        log("正在保存模型", "1")
        if (prop.getProperty("hdfs_dir") == "") {
            log("模型保存地址为空", "-1")
            return
        }
        LOG.info("保存模型")
        LOG.info(prop.getProperty("hdfs_dir") + JSONObject.fromObject(model_info).get("model_dir"))
        model.save(sc, prop.getProperty("hdfs_dir") + JSONObject.fromObject(model_info).get("model_dir") + "/" + modelid)
        //模型保存在hdfs上
        LOG.info("保存成功")

        val input_map: util.HashMap[String, String] = modelIdMap
        input_map.put("tmpParam", estC.toString)
        LOG.info(input_map.toString)
        val flg = postDataToURL(prop.getProperty("tmpparam"), input_map)
        LOG.info(flg)
        log("模型已保存", "1")
    }

    def log(msg: String, status: String) {
        val date = new Date()
        val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
        val nowtime = dateFormat.format(date)
        val log_MAP = new util.HashMap[String, String]()
        log_MAP.put("modelId", modelIdMap.get("modelId"))
        log_MAP.put("createTime", nowtime)
        log_MAP.put("status", status)
        log_MAP.put("log", msg)
        if (status != "-1")
            log_MAP.put("type", "0")
        else log_MAP.put("type", "1")

        val log_input_map = new util.HashMap[String, String]()
        log_input_map.put("key4token", "dmp")
        log_input_map.put("input", JSONObject.fromObject(log_MAP).toString)
        postDataToURL(prop.getProperty("log"), log_input_map)
    }

    def postDataToURL(url: String, params: util.HashMap[String, String]): String = {
        val conn = Jsoup.connect(url).timeout(45000).ignoreContentType(true)
        conn.data(params).post().body().text()
    }

    def postArrayToURL(modelid: String, url: String, params: JSONArray): String = {
        val input_map: util.HashMap[String, String] = modelIdMap
        input_map.put("input", JSONArray.fromObject(params).toString)
        postDataToURL(url, input_map)
    }
}