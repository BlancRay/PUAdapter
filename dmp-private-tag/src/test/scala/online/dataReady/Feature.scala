package online.dataReady

import java.util

import org.apache.commons.math3.random.RandomDataGenerator
import weka.core.Attribute


object Feature {
  var attribute: Attribute = _
  val maxValue = new util.HashMap[Int, Int]

  def setAttrNum(id: Int): Attribute = numeric(id)

  def setAttrNom(id: Int): Attribute = {
    val max = new RandomDataGenerator().nextInt(3, 50)
    val nbValues = new RandomDataGenerator().nextInt(2, max)
    nominal(id, nbValues, max)
  }

  def numeric(id: Int): Attribute = {
    new Attribute(Integer.toString(id))
  }

  def nominal(id: Int, nbValues: Int, max: Int): Attribute = {
    if (max < nbValues) {
      System.err.println("Max < nbValues, Max = " + max + ", nbValues = " + nbValues)
      System.exit(-1)
    }
    val values = new util.ArrayList[String](nbValues)
    values.add(dataGenerate.MissValue)
    val rdg = new RandomDataGenerator
    val v = rdg.nextPermutation(max, nbValues)
    for (i <- v) {
      values.add(Integer.toString(i))
    }
    new Attribute(Integer.toString(id), values)
  }

  def getAttribute(flag: Boolean, id: Int): Attribute = {
    if (flag) attribute = setAttrNom(id)
    else {
      maxValue.put(id, new RandomDataGenerator().nextPermutation(5000, 1)(0).asInstanceOf[Integer])
      attribute = setAttrNum(id)
    }
    attribute
  }

  def setAttribute(attribute: Attribute): Unit = {
    this.attribute = attribute
  }

  def getMaxValue: util.HashMap[Int, Int] = maxValue
}

