package com.zzy

import java.io.{File, PrintWriter}
import java.text.SimpleDateFormat
import java.util
import java.util.{Properties, Random}

import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.hadoop.hbase.client.{Put, Result}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableOutputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.mapreduce.Job
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object userGenerate {
  val nbUsers = 5000
  val prop = new Properties()

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "E:\\xulei\\hadoop2.6.0")
    val tagFile = "E:\\xulei\\zhiziyun\\model\\test\\tagSets.csv"
    val generate: Array[String] = dataGenerate(tagFile)
    //    generate.foreach{each=>
    //      println(each)
    //    }
    //    sys.exit(0)
    saveData2File(generate, "E:\\xulei\\zhiziyun\\model\\test\\data.txt")
    val sc = new SparkContext(new SparkConf().setAppName("put2HBase").setMaster("local[4]"))
    val path = this.getClass.getResourceAsStream("/model.properties")
    prop.load(path)
    put2hbase(generate, sc)

  }

  /**
    *
    * @param tagFile used tags
    * @return
    */
  def dataGenerate(tagFile: String): Array[String] = {
    val (tags, tagParents) = getTagsFromArray(tagFile)
    var gid = 0
    val gidTag = ArrayBuffer[String]()
    val distribute = new mutable.HashMap[String, mutable.HashMap[String, Int]]()
    while (gid < nbUsers) {
      val rdg = new RandomDataGenerator()
      val selectParents = rdg.nextPermutation(tagParents.length, new Random().nextInt(tagParents.length) + 1)
      selectParents.foreach { i =>
        val data = new StringBuffer()
        data.append(gid).append(",")
        val tagSets = tags.get(tagParents(i))
        val getI = new Random().nextInt(tagSets.get.length)
        val tagId = tagSets.get(getI)

        //        add tag recency
        val date = randomDate("2017-11-28 00:00:00", "2017-12-27 23:59:59")
        data.append(",")
        data.append("TAG_").append(tagId).append(".optime,")
          .append(date)

        //        add tag intensity
        var intensity = Double.NaN
        while (intensity.isNaN || intensity < 0 || intensity > 1) {
          intensity = roundDouble(new RandomDataGenerator().nextGaussian(0.7, 0.17), 5)
        }
        if (intensity.isNaN) {
          print("intensity = " + intensity)
          sys.exit()
        }
        data.append("TAG_").append(tagId + ",")
          .append(intensity)
        //        add in gidTag
        gidTag.append(data.toString)

        if (distribute.contains(tagParents(i))) {
          val tmp = distribute(tagParents(i))
          if (tmp.contains(tagId)) {
            tmp.put(tagId, tmp(tagId) + 1)
          } else {
            tmp.put(tagId, 1)
          }
          distribute.put(tagParents(i), tmp)
        } else {
          val tmp = new mutable.HashMap[String, Int]()
          tmp.put(tagId, 1)
          distribute.put(tagParents(i), tmp)
        }
      }
      gid += 1
    }
    tagParents.foreach { each =>
      print(each + ":[")
      tags(each).foreach { item =>
        if (tags(each).last != item)
          print(item + ":" + distribute(each)(item) + ",")
        else
          print(item + ":" + distribute(each)(item))
      }
      println("]")
    }
    //    sys.exit()
    gidTag.toArray
  }

  def getTagsFromArray(file: String): (mutable.Map[String, ArrayBuffer[String]], ArrayBuffer[String]) = {
    val tagInstances = getSourceFromFile(file)
    val tags = mutable.Map[String, ArrayBuffer[String]]()
    val tagParents = new ArrayBuffer[String]()
    tagInstances.foreach { tag =>
      val tagSplit = tag.split(",")
      val parentTag = tagSplit(2)
      val tagId = tagSplit(0)
      if (!tags.contains(parentTag)) {
        tagParents.append(parentTag)
        val tmp = new ArrayBuffer[String]()
        tmp.append(tagId)
        tags.put(parentTag, tmp)
      } else {
        val tmp: ArrayBuffer[String] = tags(parentTag)
        tmp.append(tagId)
        tags.put(parentTag, tmp)
      }
    }
    (tags, tagParents)
  }

  /**
    * 获取随机日期
    *
    * @param beginDate
    * 起始日期，格式为：yyyy-MM-dd HH:mm:ss
    * @param endDate
    * 结束日期，格式为：yyyy-MM-dd HH:mm:ss
    * @return RandomDate 随机生成的时间，格式为：yyyy-MM-dd HH:mm:ss
    */

  def randomDate(beginDate: String, endDate: String): String = {
    val format: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    val start = format.parse(beginDate); // 构造开始日期
    val end = format.parse(endDate); // 构造结束日期
    // getTime()表示返回自 1970 年 1 月 1 日 00:00:00 GMT 以来, 此 Date 对象表示的毫秒数。
    if (start.getTime >= end.getTime) {
      return null
    }
    val date = random(start.getTime, end.getTime)
    format.format(date)
  }

  def random(begin: Long, end: Long): Long = {
    val rtn: Long = begin + (Math.random() * (end - begin)).toLong
    if (rtn < begin || rtn > end) {
      return random(begin, end)
    }
    rtn
  }

  def getSourceFromFile(fileDir: String): Array[String] = {
    val file = Source.fromFile(fileDir, "utf8")
    val data = new ArrayBuffer[String]()
    var i = 0
    var firstLine = true
    for (line <- file.getLines) {
      if (firstLine) {
        firstLine = false
      }
      else {
        data.append(line)
        i += 1
      }
    }
    file.close
    data.toArray
  }

  def saveData2File(data: Array[String], filePath: String): Unit = {
    val writer = new PrintWriter(new File(filePath))
    data.foreach { each =>
      writer.println(each)
    }
    writer.close()
  }

  /**
    *
    * @param data Array 数据来源，P/U/N
    * @param sc   SparkContext
    * @return (RDD[String],RDD[LabeledPoint]) (GID,数据)
    */
  def put2hbase(data: Array[String], sc: SparkContext) {
    sc.hadoopConfiguration.set("hbase.zookeeper.property.clientPort", prop.getProperty("hbase_clientPort"))
    sc.hadoopConfiguration.set("hbase.zookeeper.quorum", prop.getProperty("hbase_quorum")) //"kylin-node4,kylin-node3,kylin-node2"
    val tableName = "TG_RESULT"
    sc.hadoopConfiguration.set(TableOutputFormat.OUTPUT_TABLE, tableName)
    val job = Job.getInstance(sc.hadoopConfiguration)
    job.setOutputKeyClass(classOf[ImmutableBytesWritable])
    job.setOutputValueClass(classOf[Result])
    job.setOutputFormatClass(classOf[TableOutputFormat[ImmutableBytesWritable]])

    val inPutRDD = sc.makeRDD(data) //构建两行记录
    val rdd = inPutRDD.map(_.split(',')).map { arr => {
      val put = new Put(Bytes.toBytes(arr(0))) //行健的值
      put.addColumn(Bytes.toBytes("info"), Bytes.toBytes(arr(1)), Bytes.toBytes(arr(2))) //info:name列的值
      put.addColumn(Bytes.toBytes("info"), Bytes.toBytes(arr(3)), Bytes.toBytes(arr(4))) //info:gender列的值
      (new ImmutableBytesWritable, put)
    }
    }
    rdd.saveAsNewAPIHadoopDataset(job.getConfiguration)
  }

  def roundDouble(value: Double, afterDecimalPoint: Int): Double = {
    val mask = Math.pow(10.0, afterDecimalPoint)
    (value * mask.round) / mask
  }
}