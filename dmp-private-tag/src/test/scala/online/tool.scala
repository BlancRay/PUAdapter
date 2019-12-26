package online

import java.text.SimpleDateFormat
import java.util
import java.util.Date

import com.fasterxml.jackson.databind.JsonNode
import com.zzy.tagModel.{LOG, mapper, modelIdMap, prop}
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
import org.jsoup.Jsoup

import scala.collection.mutable

object tool {
    def predict(points: RDD[LabeledPoint], model: RandomForestModel): RDD[Double] = {
        val numTrees = model.trees.length
        val trees = points.sparkContext.broadcast(model.trees)
        points.map { point =>
            trees.value.map(_.predict(point.features)).sum / numTrees
        }
    }

    def getAttributeInfo(modelid: String): (mutable.Map[Int, Int], mutable.Map[Int, Int]) = {
        val featureInfoJson = mapper.readTree(tool.postDataToURL(prop.getProperty("tagindex"), modelIdMap)).get("result")
        val nominalInfo = mutable.Map[Int, Int]()
        val attributeInfo = mutable.Map[Int, Int]()
        for (i <- 0 until featureInfoJson.size()) {
            val a = featureInfoJson.get(i.toString).asInt
            attributeInfo.put(i, a)
            if (a != 1)
                nominalInfo.put(i, a)
        }
        (attributeInfo, nominalInfo)
    }

    /**
      *
      * @param Type String 数据来源，P/U/N
      * @param sc   SparkContext
      * @return (RDD[String],RDD[LabeledPoint]) (GID,数据)
      */
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

        LOG.info("特征数量:" + Features.size.toString)
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
                    LOG.error(rowkey + " 的特征数为0!")
                    log(modelid, rowkey + " 的特征数为0!", "-1", prop.getProperty("log"))
                    throw new Exception(rowkey + " 的特征数为0!")
                }

                //      println(rowkey,hbaseresult)
                //从MODEL_TAG_INDEX表中读取modelID的所有标签 getTagIndexInfoByModelId

                //
                val feature = new Array[Double](factors.length)
                factors.foreach { each =>
                    if (GID_TAG_split.contains(each + 1)) {
                        feature(each) = GID_TAG_split(each + 1)
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

    /**
      * 保存模型文件和参数c
      *
      * @param model   org.apache.spark.mllib.tree.model.RandomForestModel模型
      * @param estC    Double 参数c
      * @param modelid String 模型id
      */
    def save(model: RandomForestModel, estC: Double, model_info: JsonNode, modelid: String, sc: SparkContext) {
        log(modelid, "正在保存模型", "1", prop.getProperty("log"))
        if (prop.getProperty("hdfs_dir") == "") {
            log(modelid, "模型保存地址为空", "-1", prop.getProperty("log"))
            return
            //      sys.exit(-1)
        }
        LOG.info("保存模型")
        println(model_info.get("model_dir").asText)
        model.save(sc, prop.getProperty("hdfs_dir") + model_info.get("model_dir").asText + "/" + modelid)
        //模型保存在hdfs上
        LOG.info("保存成功")

        val input_map: util.HashMap[String, String] = modelIdMap
        input_map.put("tmpParam", estC.toString)
        LOG.info(input_map.toString)
        val flg = postDataToURL(prop.getProperty("tmpparam"), input_map)
        LOG.info(flg)
        log(modelid, "模型已保存", "1", prop.getProperty("log"))
    }

    def log(modelid: String, msg: String, status: String, url: String) {
        val date = new Date()
        val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss")
        val nowtime = dateFormat.format(date)
        val log_MAP = new util.HashMap[String, String]()
        log_MAP.put("modelId", modelid)
        log_MAP.put("createTime", nowtime)
        log_MAP.put("status", status)
        log_MAP.put("log", msg)
        if (status != "-1")
            log_MAP.put("type", "0")
        else log_MAP.put("type", "1")

        val log_input_map = new util.HashMap[String, String]()
        log_input_map.put("key4token", "dmp")
        log_input_map.put("input", mapper.writeValueAsString(log_MAP))
        postDataToURL(url, log_input_map)
    }

    def postDataToURL(url: String, params: util.HashMap[String, String]): String = {
        val conn = Jsoup.connect(url).timeout(45000).ignoreContentType(true)
        conn.data(params).post().body().text()
    }

    def postArrayToURL(modelid: String, url: String, params: JsonNode): String = {
        val input_map: util.HashMap[String, String] = modelIdMap
        input_map.put("input", mapper.writeValueAsString(params))
        //    println(input_map)
        postDataToURL(url, input_map)
    }
}