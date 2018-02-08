package online.dataReady

import java.util.Properties

import org.apache.hadoop.hbase.client.{Put, Result}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableOutputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.mapreduce.Job
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object upload2HBase {
  val prop = new Properties()

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
    val sc = new SparkContext(new SparkConf().setAppName("put2HBase").setMaster("local[4]"))
    val path = this.getClass.getResourceAsStream("/model.properties")
    prop.load(path)
    //    var generate=readFile(dataGenerate.dir+"test.txt")
    //    put2hbase(generate, sc,"P")
    var generate = readFile(dataGenerate.dir + "train_P_label")
    put2hbase(generate, sc, "P")
    generate = readFile(dataGenerate.dir + "train_U")
    put2hbase(generate, sc, "U")
    generate = readFile(dataGenerate.dir + "test_All")
    put2hbase(generate, sc, "N")
  }

  /**
    *
    * @param data Array 数据来源，P/U/N
    * @param sc   SparkContext
    * @return (RDD[String],RDD[LabeledPoint]) (GID,数据)
    */
  def put2hbase(data: Array[String], sc: SparkContext, dataType: String) {
    sc.hadoopConfiguration.set("hbase.zookeeper.property.clientPort", prop.getProperty("hbase_clientPort"))
    sc.hadoopConfiguration.set("hbase.zookeeper.quorum", prop.getProperty("hbase_quorum")) //"kylin-node4,kylin-node3,kylin-node2"
    val tableName = if (dataType.contains("P")) prop.getProperty("p_source_table") else if (dataType.contains("N")) prop.getProperty("n_source_table") else prop.getProperty("u_source_table")
    sc.hadoopConfiguration.set(TableOutputFormat.OUTPUT_TABLE, tableName)
    val job = Job.getInstance(sc.hadoopConfiguration)
    job.setOutputKeyClass(classOf[ImmutableBytesWritable])
    job.setOutputValueClass(classOf[Result])
    job.setOutputFormatClass(classOf[TableOutputFormat[ImmutableBytesWritable]])

    val inPutRDD = sc.makeRDD(data) //构建两行记录
    val rdd = inPutRDD.map(_.split(',')).map {
      arr => {
        val put = new Put(Bytes.toBytes(arr(0))) //行健的值
        put.addColumn(Bytes.toBytes("info"), Bytes.toBytes("trait"), Bytes.toBytes(arr(1))) //info:name列的值
        (new ImmutableBytesWritable, put)
      }
    }
    rdd.saveAsNewAPIHadoopDataset(job.getConfiguration)
  }

  def readFile(filename: String): Array[String] = {
    val data = new ArrayBuffer[String]()
    val readFile = Source.fromFile(filename).getLines
    readFile.foreach {
      line =>
        val newline = line.replace("\n", "")
        data.append(newline)
    }
    data.toArray
  }
}
