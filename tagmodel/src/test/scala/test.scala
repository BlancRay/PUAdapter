import java.util

import net.sf.json.JSONObject
import org.jsoup.Jsoup

object test {
  def main(args: Array[String]): Unit = {
    try {
      val log_MAP = new util.HashMap[String, String]()
      log_MAP.put("modelId", "asdf")
      log_MAP.put("createTime", "asdf")
      val log_input_map = new util.HashMap[String, String]()
      log_input_map.put("key4token", "dmp")
      log_input_map.put("input", JSONObject.fromObject(log_MAP).toString)
      val conn = Jsoup.connect("http://192.168.0.251:669/dmp/queryInterface/insertModelTrainLog.action").timeout(45000).ignoreContentType(true)
      conn.data(log_input_map).post().body().text()
      val code = conn.execute().statusCode()
      println(code)
    }
    catch {
      case e: Throwable =>
        println(e.printStackTrace())
    }
  }
}
