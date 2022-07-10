#!/usr/bin/env sh
mvn clean package

spark-submit --class azhu.AppSQL --master "local[*]" target/azhu_project-1.0-SNAPSHOT.jar data/Tweets_10k.json
