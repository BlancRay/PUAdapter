import java.io.File

import scala.util.Random

object codeTest {
  def main(args: Array[String]): Unit = {
    val random = Random.nextDouble()
    if (random > 0.5) {
      println(random)
      throw new Exception(random + ">0.5")
    } else
      println(random)
    println("222222")
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
}
