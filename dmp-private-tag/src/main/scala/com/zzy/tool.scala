package com.zzy

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

import scala.collection.JavaConversions._
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

    def getAttributeInfo(modelid: String): (mutable.Map[Int, Int], mutable.Map[Int, Int], mutable.Map[Int, String], mutable.Map[String, Int]) = {
        val result = tool.postDataToURL(prop.getProperty("traitindex"), modelIdMap)
        LOG.info(result)
        val featureInfoJson = mapper.readTree(result).get("result")
        val nominalInfo = mutable.Map[Int, Int]() //nominal attribute,#values, for model fit use
        val attributeInfo = mutable.Map[Int, Int]() //attribute,#values
        val traitIDIndex = mutable.Map[String, Int]() //traitID,index
        var i = 0
        featureInfoJson.foreach { each =>
            val trait_id = each.get("traitId").asText
            if (!traitIDIndex.contains(trait_id)) {
                traitIDIndex.put(trait_id, i)
                attributeInfo.put(i, 1)
                if (each.get("flag").asText == "1") { //离散特征 flag=1
                    nominalInfo.put(i, 1)
                }
                i = i + 1
            } else {
                attributeInfo.update(traitIDIndex(trait_id), attributeInfo(traitIDIndex(trait_id)) + 1)
                if (each.get("flag").asText == "1") {
                    nominalInfo.update(traitIDIndex(trait_id), nominalInfo(traitIDIndex(trait_id)) + 1)
                }
            }
        }
        val attributeIndex = mutable.Map[Int, String]() //index,traitID
        traitIDIndex.foreach { each =>
            attributeIndex.put(each._2, each._1)
        }

        (attributeInfo, nominalInfo, attributeIndex, traitIDIndex)
    }

    /**
      *
      * @param Type String 数据来源，P/U/N
      * @param sc   SparkContext
      * @return (RDD[String],RDD[LabeledPoint]) (GID,数据)
      */
    def read_convert(modelid: String, Type: String, sc: SparkContext, attributeInfo: mutable.Map[Int, Int], traitIDIndex: mutable.Map[String, Int]): (RDD[String], RDD[LabeledPoint]) = {
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
        }
        //        val gid = new Array[String](count.toInt)
        //        val lp = new Array[LabeledPoint](count.toInt)
        //        var i = 0

        LOG.info("特征数量:" + attributeInfo.size.toString)
        val factors = Array.range(0, attributeInfo.size)
        val result = res.map { res_each =>
            val gid = res_each._1
            val hbaseresult = res_each._2
            val GID_TAG_split = new mutable.HashMap[Int, Double]()
            if (hbaseresult != "") {
                val GID_TAG_SET = hbaseresult.split(";")
                GID_TAG_SET.foreach { each =>
                    val tag_split = each.split(":")
                    if (attributeInfo(traitIDIndex(tag_split(0))) == 1) //根据特征的value数量判断是否是离散型,数量为1，则为连续型
                        GID_TAG_split.put(traitIDIndex(tag_split(0)), tag_split(1).toDouble)
                    else
                        GID_TAG_split.put(traitIDIndex(tag_split(0)), tag_split(1).toInt)
                }
            } else {
                LOG.error(gid + " 的特征数为0!")
                log(gid + " 的特征数为0!", "-1")
                throw new Exception(gid + " 的特征数为0!")
            }

            //      println(rowkey,hbaseresult)
            //从MODEL_TAG_INDEX表中读取modelID的所有标签 getTagIndexInfoByModelId

            //
            val feature = new Array[Double](factors.length)
            factors.foreach { each =>
                if (GID_TAG_split.contains(each)) {
                    feature(each) = GID_TAG_split(each)
                } else {
                    if (attributeInfo(each) == 1) //连续变量长度为1
                        feature(each) = -1.0
                    else //离散变量长度至少为2(包含一个"NULL")
                        feature(each) = 0.0
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
    def save(model: RandomForestModel, estC: Double, model_info: JsonNode, modelid: String, sc: SparkContext) {
        LOG.info("保存模型")
        log("正在保存模型", "1")

        if (prop.getProperty("hdfs_dir") == "") {
            log("模型保存地址为空", "-1")
            return
            //      sys.exit(-1)
        }

        LOG.info("模型保存在：" + model_info.get("modelDir").asText)

        model.save(sc, prop.getProperty("hdfs_dir") + model_info.get("modelDir") + "/" + modelid)

        LOG.info("模型文件保存成功")

        val input_map: util.HashMap[String, String] = modelIdMap
        input_map.put("tmpParam", estC.toString)

        LOG.info("保存参数c" + input_map.toString)

        postDataToURL(prop.getProperty("tmpparam"), input_map)

        log("训练模型已保存", "1")
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
        log_input_map.put("input", mapper.writeValueAsString(log_MAP))
        postDataToURL(prop.getProperty("log"), log_input_map)
    }

    def postDataToURL(url: String, params: util.HashMap[String, String]): String = {
        val conn = Jsoup.connect(url).timeout(45000).ignoreContentType(true)
        conn.data(params).post().body().text()
    }

    def postArrayToURL(url: String, params: JsonNode): String = {
        val input_map: util.HashMap[String, String] = modelIdMap
        input_map.put("input", mapper.writeValueAsString(params))
        //            println(input_map)
        postDataToURL(url, input_map)
    }
}