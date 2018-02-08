package online.dataReady

import java.io._
import java.util
import java.util.Random

import org.apache.commons.math3.random.RandomDataGenerator
import weka.core.{Attribute, Instances, Utils}

/**
  * @author xulei
  *
  */
object dataGenerate {
  val MissValue = "?"
  var n = 2000 // 标签数
  var DS = 10 // 种子数量
  var DP = 1000 // 正例数量
  var DN = 10000 // 未标记数量
  // public static int m = DP + DU;// 样本总数
  var Pr1 = 0.5 //正例数据遗传率
  var Pr2 = 0.5 //负例数据遗传率
  val dir = "E:\\xulei\\zhiziyun\\model\\test\\test\\"
  val sFile = new File(dir + "Sdata.arff")
  val attrInfoFile = new File(dir + "attrinfo.txt")
  val maxValueFile = new File(dir + "maxValue.txt")
  val pFile = new File(dir + "Pdata")
  val nFile = new File(dir + "Ndata")
  val uFile = new File(dir + "Udata")
  var attributes: util.ArrayList[Attribute] = _


  def main(args: Array[String]): Unit = {
    val mode = "generate"
    //    val mode = "read"
    if (mode eq "generate") {
      attributes = tool.defAttr()
      val nbattrs = tool.nbAttrs(attributes)
      var infoWriter = new PrintWriter(new FileOutputStream(attrInfoFile), true)
      val nbattrs_local = nbattrs.toArray.sortBy(_._1)
      for (each <- nbattrs_local) {
        infoWriter.write(each._1 + "\t" + each._2 + "\n")
      }
      infoWriter.close()
      infoWriter = new PrintWriter(new FileOutputStream(maxValueFile), true)
      infoWriter.write(tool.maxValue.toString.replace(", ", "\n").replace("=", "\t").replace("{", "").replace("}", ""))
      infoWriter.close()
      val seeds = GenerateSeed(sFile)
      val attrJSON = new StringBuffer()
      for (i <- 0 until seeds.numAttributes()) {
        if (i != 0) {
          attrJSON.append(";")
        }
        if (seeds.attribute(i).isNominal) {
          attrJSON.append(i).append(":").append(seeds.attribute(i).numValues())
        } else {
          attrJSON.append(i).append(":").append(1)
        }
      }
      val writer = new PrintWriter(new FileOutputStream(dir + "AttrbuiteJSON.txt"), true)
      writer.print(attrJSON)
      writer.close()
      Generate(new Instances(seeds), pFile, DP, Pr1)
      Generate(new Instances(seeds), nFile, DN, Pr2)
      //      Generate(new Instances(seeds), uFile, DP, 0.5)
    } else {
      val seeds = readSeedFromFile(sFile, maxValueFile)
      val attrJSON = new StringBuffer
      for (i <- 0 until seeds.numAttributes) {
        if (i != 0) attrJSON.append(";")
        if (seeds.attribute(i).isNominal)
          attrJSON.append(i).append(":").append(seeds.attribute(i).numValues)
        else attrJSON.append(i).append(":").append(1)
      }
      val writer = new PrintWriter(new FileOutputStream(dir + "AttrbuiteJSON.txt"), true)
      writer.print(attrJSON)
      writer.close()
      Generate(new Instances(seeds), pFile, DP, Pr1)
      Generate(new Instances(seeds), nFile, DN, Pr2)
      // Generate(new Instances(seeds), uFile, DN,Pr2)
    }
  }

  def GenerateSeed(filename: File): Instances = {
    println("Generating Seed samples...")
    val seeds = tool.seed(attributes)
    println("Generate finished!")
    for (i <- 0 until seeds.size()) {
      val nbset2null = new Array[Double](2)
      for (j <- 0 until n) {
        if (seeds.get(i).attribute(j).isNominal && (seeds.get(i).value(j) == 0))
          nbset2null(0) += 1
        else if (seeds.get(i).attribute(j).isNumeric && (seeds.get(i).value(j) == -1))
          nbset2null(1) += 1
      }
      println(util.Arrays.toString(nbset2null) + Utils.sum(nbset2null) / n)
    }
    // 生成种子个体并写入文件
    val seedWriter = new PrintWriter(new FileOutputStream(filename), true)
    seedWriter.write(seeds.toString)
    seedWriter.close()
    seeds
  }

  def Generate(seeds: Instances, filename: File, nbinstances: Int, percent: Double) {
    println("Generate Samples...")
    var selectasseed = new Array[Int](0)
    do {
      selectasseed = new RandomDataGenerator().nextPermutation(seeds.size, new RandomDataGenerator().nextInt(1, seeds.size))
    } //if random
    while (selectasseed.length < 3)
    util.Arrays.sort(selectasseed)
    println(util.Arrays.toString(selectasseed).replace("[", "").replace("]", "").replace(" ", ""))
    // 生成种子人群，并写入文件中
    val whichseed = new Array[Int](10)
    val sBuffer = new StringBuffer()
    val flagBuffer = new StringBuffer()
    val dataBuffer = new StringBuffer()
    for (i <- 0 until nbinstances) {
      val flag = selectasseed((Math.random * selectasseed.length).toInt) // 随机选择一个种子个体
      flagBuffer.append(flag).append("\n")
      whichseed(flag) += 1 // 统计种子个体被选取次数
      val gid_i = seeds.get(flag)
      // System.out.println(gid_i.toString());
      var first = true
      sBuffer.append("a_" + i).append(",")
      for (j <- 0 until gid_i.numAttributes()) {
        var value = 0
        var continue = true
        if (gid_i.attribute(j).isNumeric) { // 连续变量
          if ((gid_i.value(j) != -1) && new Random().nextDouble > percent) {
            gid_i.setValue(j, -1) // 置空
            value = -1
          } else if ((gid_i.value(j) == -1) && new Random().nextDouble > percent) { //随机生成
            val max = tool.maxValue.get(j)
            value = new RandomDataGenerator().nextInt(0, max)
            gid_i.setValue(j, value)
          } else
            value = gid_i.value(j).asInstanceOf[Int]
          if (value == -1)
            continue = false
        } else {
          if ((gid_i.value(j) != 0) && new Random().nextDouble > percent) { // 离散变量
            gid_i.setValue(j, MissValue)
            value = attributes.get(j).indexOfValue(MissValue)
          } else if ((gid_i.value(j) == 0) && new Random().nextDouble > percent) { // 随机生成
            value = new RandomDataGenerator().nextInt(1, gid_i.attribute(j).numValues - 1)
            gid_i.setValue(j, gid_i.attribute(j).value(value))
          } else
            value = gid_i.attribute(j).indexOfValue(gid_i.stringValue(j))
          if (value == 0) { // testol
            continue = false
          }
        }
        if (continue) {
          if (first) {
            sBuffer.append(j + ":" + value)
            first = false
          } else
            sBuffer.append(";" + j + ":" + value)
        }
      }
      //			System.out.println(gid_i.toString());
      //			System.out.println(sBuffer);
      //			System.exit(0);
      sBuffer.append("\n")
      dataBuffer.append(gid_i).append("\n")
    }
    var writer = new PrintWriter(new FileOutputStream(filename), true)
    writer.print(sBuffer)
    writer.close()
    writer = new PrintWriter(new FileOutputStream(filename + "_arff"), true)
    writer.println(dataBuffer)
    writer = new PrintWriter(new FileOutputStream(filename + "_id"), true)
    writer.print(flagBuffer)
    writer.close()
    System.out.println(util.Arrays.toString(whichseed))
  }

  def readSeedFromFile(strARFFFileName: File, maxvalue: File): Instances = {
    val seed = getARFFDatasetFromFile(strARFFFileName)
    tool.readHashMap(maxvalue)
    attributes = new util.ArrayList[Attribute]
    val en = seed.enumerateAttributes
    while ( {
      en.hasMoreElements
    }) {
      val attr = en.nextElement
      attributes.add(attr)
    }
    println("Generate finished!")
    seed
  }

  def getARFFDatasetFromFile(strARFFFileName: File) = new Instances(new BufferedReader(new FileReader(strARFFFileName)))
}
