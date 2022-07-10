package azhu

import org.apache.spark.sql.functions.{array_intersect,explode, col, lit}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{SparkSession}


object AppSQL {

  def task(inFile: String){

    val outFile = "tweets_clean"

    val conf = new SparkConf().setAppName("TwitterAnalysis")
    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val sparkSession: SparkSession.Builder = SparkSession.builder().config(conf)
    val spark: SparkSession = sparkSession.getOrCreate()
    import spark.implicits._

    val tweets = spark.read.json(inFile)
    var new_tweets = tweets.select(
      col("id"),
      col("text"),
      col("entities.hashtags.text").alias("hashtags"),
      col("user.description").alias("user_description"),
      col("retweet_count"),
      col("reply_count"),
      col("quoted_status_id"))

    // write the cleaned data in json
    new_tweets.coalesce(1).write.format("json").save(outFile)

    // get top 20 most frequent hashtags
    val tags = new_tweets.select(explode($"hashtags"))
    val tagsTable = tags.createOrReplaceTempView("table")
    val keywords = spark.sql("SELECT DISTINCT col, COUNT(col) AS cnt FROM table GROUP BY col ORDER by cnt DESC LIMIT 20")
    keywords.show()

    val keywordList = keywords.select("col").as[String].collect

    //    Task2
    new_tweets = new_tweets.withColumn("temp", lit(keywordList))
    new_tweets = new_tweets.withColumn("topic",array_intersect(col("hashtags"), col("temp")))

    new_tweets = new_tweets.drop(col("temp")).drop(col("hashtags"))

    new_tweets.createOrReplaceTempView("tweetsTemp")
    val query: String = s"""SELECT * FROM tweetsTemp WHERE size(topic) > 0"""
    new_tweets = spark.sql(query)

    new_tweets = new_tweets.withColumn("topic", col("topic").getItem(0))

    print("Total topic data for 10k data: " + new_tweets.count() + "\n")
    new_tweets.coalesce(1).write.format("json").save("./tweets_topic")

  }


  def main(args: Array[String]) {

    val inFile = args(0)
    task(inFile)

  }
}
