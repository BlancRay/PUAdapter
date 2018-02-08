package online

import java.io.{File, FileOutputStream, PrintWriter}
import java.util.Properties

import net.sf.json.JSONObject
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.{ConnectionFactory, Scan}
import org.apache.hadoop.hbase.filter.PrefixFilter
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.protobuf.ProtobufUtil
import org.apache.hadoop.hbase.util.{Base64, Bytes}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.io.Source

object unit_train {
    val prop = new Properties()

    def main(args: Array[String]): Unit = {
        System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
        val path = this.getClass.getResourceAsStream("../model.properties")
        prop.load(path)
        val sc = new SparkContext(new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]"))
        val json = "{" + Source.fromFile(online.dataReady.dataGenerate.dir + "AttrbuiteJSON.txt").getLines().next() + "}"
        val featureInfoJson = JSONObject.fromObject(json)
        val nominalInfo = mutable.Map[Int, Int]()
        val attributeInfo = mutable.Map[Int, Int]()
        for (i <- 0 until featureInfoJson.size()) {
            val a = featureInfoJson.getInt(i.toString)
            attributeInfo.put(i, a)
            if (a != 1)
                nominalInfo.put(i, a)
        }
        val (_, p) = read_convert("a", "P_SOURCE", sc, attributeInfo)
        //读取数据时需要所有特征信息
        val (_, u) = read_convert("a", "U_SOURCE", sc, attributeInfo)
        val (estC, model) = fit(p, u, nominalInfo.toMap)
        //训练模型时，只需要nominal类型特征信息
        val saveC = new PrintWriter(new FileOutputStream(dataReady.dataGenerate.dir + "estC"))
        saveC.write(estC.toString)
        saveC.close()
        val modelFile = new File(dataReady.dataGenerate.dir + "model")
        dirDel(modelFile)
        model.save(sc, dataReady.dataGenerate.dir + "model")
    }

    def dirDel(path: File) {
        if (!path.exists())
            return
        else if (path.isFile) {
            path.delete()
            println(path + ":  文件被删除")
            return
        }
        val file: Array[File] = path.listFiles()
        for (d <- file) {
            dirDel(d)
        }
        path.delete()
        println(path + ":  目录被删除")
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

    def fit(POS: RDD[LabeledPoint], UNL: RDD[LabeledPoint], categoryInfo: Map[Int, Int]): (Double, RandomForestModel) = {
        val hold_out_ratio = 0.2
        var c = Double.NaN
        var model_hold_out: RandomForestModel = null
        val splits = POS.randomSplit(Array(hold_out_ratio, 1.0 - hold_out_ratio))
        val (p_test, p_train) = (splits(0), splits(1))
        // Train a RandomForest model.
        val trainData = p_train.union(UNL)

        model_hold_out = RandomForest.trainClassifier(trainData, 2, categoryInfo, 50, "auto", "gini", 25, 200)
        val hold_out_predictions = tool.predict(p_test, model_hold_out)
        c = hold_out_predictions.sum() / hold_out_predictions.count()
        if (c.isNaN) {
            println("C is Nan")
        }
        (c, model_hold_out)
    }
}
