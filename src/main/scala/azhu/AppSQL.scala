package azhu

import org.apache.spark.sql.functions.{array_intersect,explode, col, lit}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

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


    // Task 3
    println("Beginning topic prediction model building")
    var tweetsDF = sparkSession.read.json(inFile)

    // combine text columns
    tweetsDF = tweetsDF.withColumn("all_text",concat_ws(" ",col("text"),col("user_description")))

    val indexer = new StringIndexer()
      .setInputCol("topic")
      .setOutputCol("label")
      .fit(tweetsDF)

    val tokenizer = new Tokenizer() // tokenize all text
      .setInputCol("all_text")
      .setOutputCol("words")

    val word2vec = new Word2Vec() // word2vec instead of TF.IDF - can use seed for init
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
      .setVectorSize(300)

    val lr = new LogisticRegression() // Using logistic reg as classifier
      .setMaxIter(10)

    val pipeline = new Pipeline() // pipeline
      .setStages(Array(indexer,tokenizer,word2vec,lr))

    val paramGrid = new ParamGridBuilder() // parameter grid for cross validation
      .addGrid(word2vec.minCount, Array(0, 1))
      .addGrid(lr.elasticNetParam, Array(0,0.01,0.1,0.3,0.8))
      .addGrid(lr.regParam, Array(0,0.01,0.1,0.3,0.8))
      .build()

    val cv = new CrossValidator() // Running cross validator to produce best model given parameter grid
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)  // Use 3+ in practice
      .setParallelism(3)  // Evaluate up to 3 parameter settings in parallel docs say ~10

    val Array(training, test) = tweetsDF.randomSplit(Array(0.7, 0.3)) // splits data into training and test - can use seed
    val model = cv.fit(training) // build model with training split
    val minCount = model.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[Word2VecModel].getMinCount
    val elasticRegParam = model.bestModel.asInstanceOf[PipelineModel].stages(3).asInstanceOf[LogisticRegressionModel].getElasticNetParam
    val RegParam = model.bestModel.asInstanceOf[PipelineModel].stages(3).asInstanceOf[LogisticRegressionModel].getRegParam

    model.write.overwrite().save("models/logistic-regression-model") // save model

    val results = model.transform(test) // use model on test data
      .select("id", "text", "topic", "user_description", "label", "prediction")

    // get and print out metrics
    val multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
    val metrics = multiclassClassificationEvaluator.getMetrics(results)
    println("Summary Statistics")
    println(s"Accuracy = ${metrics.accuracy}")
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}\n")
    println("Best model parameters")
    println(s"Best minCount: $minCount")
    println(s"Best elasticRegParam: $elasticRegParam")
    println(s"Best RegParam: $RegParam")  
  }


  def main(args: Array[String]) {

    val inFile = args(0)
    task(inFile)

  }
}
