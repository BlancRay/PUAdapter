import java.util.Properties

import com.fasterxml.jackson.databind.ObjectMapper
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.{ConnectionFactory, Scan}
import org.apache.hadoop.hbase.filter.PrefixFilter
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.protobuf.ProtobufUtil
import org.apache.hadoop.hbase.util.{Base64, Bytes}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.io.Source

object codeTest {
    val prop = new Properties()

    def main(args: Array[String]): Unit = {
        System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
        val path = this.getClass.getResourceAsStream("/model.properties")
        prop.load(path)
        val sc = new SparkContext(new SparkConf().setAppName("RandomForestClassificationTrain").setMaster("local[4]"))
        val AttrbuiteJSONFile = Source.fromFile(online.dataReady.dataGenerate.dir + "AttrbuiteJSON.txt")
        val categoryInfoJson = new ObjectMapper().readTree("{" + AttrbuiteJSONFile.getLines.mkString + "}")
        AttrbuiteJSONFile.close()
        val categoryInfo = mutable.Map[Int, Int]()
        for (i <- 0 until categoryInfoJson.size()) {
            val a = categoryInfoJson.get(i.toString).asInt()
            categoryInfo.put(i, a)
        }
        val (_, p) = read_convert("a", "P_SOURCE", sc, categoryInfo)
        p.collect().foreach { e =>
            print(e.features)
        }
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
                    if (GID_TAG_split.contains(each)) { //若数据库中下标从1开始,则为contains(each+1),否则为contains(each)
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
