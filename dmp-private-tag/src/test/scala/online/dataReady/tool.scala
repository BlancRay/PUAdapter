
package online.dataReady

import java.io.File
import java.util
import java.util.Random

import org.apache.commons.math3.random.RandomDataGenerator
import weka.core.{Attribute, DenseInstance, Instances}

import scala.collection.mutable
import scala.io.Source


object tool {
  var randGen: RandomDataGenerator = _
  var maxValue = new util.HashMap[Int, Int]

  def defAttr(): util.ArrayList[Attribute] = {
    val attributes = new util.ArrayList[Attribute]()
    for (i <- 0 until dataGenerate.n) {
      val flag = new Random().nextBoolean
      attributes.add(Feature.getAttribute(flag, i))
      maxValue.putAll(Feature.getMaxValue)
    }
    attributes
  }

  def seed(attributes: util.ArrayList[Attribute]): Instances = {
    val instances = new Instances("sample", attributes, 0)
    for (i <- 0 until dataGenerate.DS) {
      val instance = new DenseInstance(attributes.size)
      instance.setDataset(instances)
      for (j <- 0 until attributes.size()) {
        if (attributes.get(j).isNumeric) { // 连续型
          if (maxValue.get(j) < 1) {
            instance.setValue(j, 0)
            System.err.println("attributes " + j + " value " + maxValue.get(j))
          }
          else if (new Random().nextDouble > 0.2) {
            val value = new RandomDataGenerator().nextPermutation(maxValue.get(j), 1)(0)
            instance.setValue(j, value)
          }
          else instance.setValue(j, -1)
        }
        else { //离散型
          if (new Random().nextDouble > 0.2) {
            val value = attributes.get(j).value(new RandomDataGenerator().nextPermutation(attributes.get(j).numValues, 1)(0))
            instance.setValue(j, value)
          }
          else instance.setValue(j, dataGenerate.MissValue)
        }
      }
      instances.add(instance)
    }
    instances
  }

  def nbAttrs(attributes: util.ArrayList[Attribute]): mutable.HashMap[Int, Int] = {
    val attrInfo = new mutable.HashMap[Int, Int]()
    for (i <- 0 until attributes.size()) {
      if (attributes.get(i).isNominal)
        attrInfo.put(i, attributes.get(i).numValues)
    }
    attrInfo
  }

  def readHashMap(filename: File): Unit = {
    val hashMap = new util.HashMap[Int, Int]
    val readFile = Source.fromFile(filename).getLines
    readFile.foreach { line =>
      val kv = line.split("\t")
      hashMap.put(kv(0).toInt, kv(1).toInt)
    }
    this.maxValue = hashMap
  }
}
