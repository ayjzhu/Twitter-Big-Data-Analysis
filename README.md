# Twitter-Big-Data-Analysis
Big Data Analysis using Spark SQL with Scala on a 100k twitter data set


## Overview
In this project, Twitter Data Analysis breaks it into 3 different tasks: In the first task, the data preparation 1 which involves data cleaning on a 10kpoints twitter dataset to get the relevant attributes and store the output in a new JSON. In addition, run a top-k SQL query to select the top 20 most frequent hashtags keywords on the clean data. In the second task, the data preparation 2 is to add a new column topic, that will include the intersection of the hashtag and top 20 keywords. Task 3 invovles with topic prediction which to build a machine learning model that assigns a topic for each tweet based on the classified tweets. The model should learn the relationship between all features and the topic. Then, it applies this model to all data to predict one topic for each tweet. The machine learning pipeline should include the following.

Both task 1 and task 2 use Spark SQL, because it provides a mix of SQL queries with Spark and
it can easily run interactive queries via API calls such as Scala, Java, Python, etc.

## Methods
### Spark SQL
Spark is made for large data analysis and meets the need for the project. Spark DataFrame API provides the necessary tools to parse, process, and store the tweets into a dataframe for further analysis.

### Spark MLlib
Spark’s MLlib provides transformers, estimators, and validators used to build the classification
model.

## Tasks & Results
### Task 1 Data preparation 1:
This task involves loading the raw data from the 10k points JSON file, extracting the relevant attributes, and storing the output to the new JSON file called “tweets_clean”. Lastly, run a top-k SQL query on the cleaned data to get the top 20 most frequent hashtags using the explode function to first produce a list of all hashtags and perform the count query and collect the result in an array of keywords.

The relevant attributes are below:
`id, text, entities.hashtags.text, user.description, retweet_count, reply_count, and quoted_status_id`

### Task 2 Data preparation 2:
This task continues with the list of top 20 hashtags obtained from task 1 and compares with each tweet to indicate whether it's a topic and add it as a new column to the data. Using the array_intersect function to compute the intersection between the list of hashtags and the list of the most frequent keywords. Lastly, keep only the records that have a topic and store the output in a JSON file named “tweets_topic”.

Total topic data (# of tweet contains the top 20 keywords in their hashtags value) for 10k data: `269`

### Task 3 Topic prediction:
The model is built using a pipeline composed of transformers: string indexer, text tokenizer, and word2vec with logistic regression as the estimator. A parameter grid testing the word2vec minCounts and logistic regression’s regularization parameters is used with the cross-validator to produce the best model.
