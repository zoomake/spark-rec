package org.pact518

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

object moviesALS {

  def parseRating(str: String): (Long, Rating) = {
    val fields = str.split(",")
    assert(fields.size == 4)
    (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat))
  }

  def main(args: Array[String]): Unit = {
    val dir = "C:\\Users\\chinaso\\Desktop\\movielens\\ml-20m"
    val conf = new SparkConf().setAppName("ALSReco").setMaster("local")
    val spark = SparkSession.builder.config(conf).getOrCreate
    val sc = spark.sparkContext
    val ratings = sc.textFile(dir + "\\ratings.csv").map(parseRating)
    val movies = sc.textFile(dir + "\\movies.csv").map {
      line =>
        val fields = line.split(",")
        (fields(0).toInt, fields(1))
    }.collect().toMap

    val numRatings = ratings.count  //评分总数
    val numUsers = ratings.map(_._2.user).distinct.count  //用户总数
    val numMovies = ratings.map(_._2.product).distinct.count   //电影总数

    //提取一个得到最多评分的电影子集，以便进行评分启发
    val mostRatedMovieIds = ratings.map(_._2.product)  //根据条件获取电影id
      .countByValue //count统计RDD中元素的个数,返回int;  countByKey以key为单位进行统计,返回map;  countByValue统计RDD中各个value的出现次数,返回一个map。
      .toSeq        //转换成Seq   Seq是列表，适合存有序重复数据，进行快速插入/删除元素等场景   Set是集合，适合存无序非重复数据，进行快速查找海量元素的等场景
      .sortBy(-_._2)
      .take(50)
      .map(_._1)

    val random = new Random(0)

    val selectedMovies = mostRatedMovieIds.filter(x => random.nextDouble() < 0.2) //获取电影ID，电影名称 Seq
      .map(x => (x, movies(x)))
      .toSeq

    val myRatings = elicitateRatings(selectedMovies)  //对热门电影进行评分
    val myRatingsRDD = sc.parallelize(myRatings, 1)
    val numPartitions = 20

    val training = ratings.filter(x => x._1 < 6)
      .values
      .union(myRatingsRDD)
      .repartition(numPartitions)
      .persist

    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .union(myRatingsRDD)
      .repartition(numPartitions)
      .persist

    val test = ratings.filter(x => x._1 >= 8)
      .values
      .union(myRatingsRDD)
      .repartition(numPartitions)
      .persist

    val numTraining = training.count
    val numValidation = validation.count
    val numTest = test.count

    val ranks = List(8, 12) //隐藏因子的个数
    val lambdas = List(1, 11) //正则项的惩罚系数
    val numIters = List(10, 20) //迭代次数

    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -0.1
    var bestNumIter = -1

    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(training, rank, numIter, lambda)
      val validationRmse = computeRmse(model, validation, numValidation)
      println("RMSE=" + validationRmse + "rank= " + rank + "lambda=" + lambda + "numIter=" + numIter)
      if(validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    //在测试集上评估得到的最佳模型
    val testRmse = computeRmse(bestModel.get, test, numTest)
    println("testRmse：" + testRmse)

    //设置朴素基线并与最佳模型比较
    val maenRating = training.union(validation).map(_.rating).mean
    val bestlineRmse = math.sqrt(test.map(x => (maenRating - x.rating) * (maenRating - x.rating)).reduce(_+_)/numTest)
    val improvement = (bestlineRmse - testRmse) / bestlineRmse * 100
    println("improvement:" + "%1.2f".format(improvement))

    //产生个性化推荐
    val myRatedMovieIds = myRatings.map(_.product).toSet
    val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
    val recommendations = bestModel.get.predict(candidates.map((0, _))).collect.sortBy(-_.rating).take(50)

    var i = 1
    println("recommend for you:")
    recommendations.foreach { r =>
      println("%2d".format(i) + ":" + movies(r.product))
      i += 1
    }
    sc.stop()
  }

  def elicitateRatings(movies: Seq[(Int, String)]) = {
    //    println("Please input:")
    val ratings = movies.flatMap { x =>
      var rating: Option[Rating] = None
      var valid = false
      while (!valid) {
        println(x._2 + ":")
        try {
          val r = Console.readFloat()
          if (r < 0 || r > 5) {
            println("anew")
          } else {
            valid = true
            if (r > 0) {
              rating = Some(Rating(0, x._1, r))
            }
          }
        } catch {
          case e: Exception => println()
        }
      }
      rating match {
        case Some(r) => Iterator(r)
        case None => Iterator.empty
      }
    }
    if (ratings.isEmpty) {
      error("")
    } else {
      ratings
    }
  }

  def computeRmse(model: MatrixFactorizationModel, data:RDD[Rating], n:Long) = {
    val predictions : RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    data.foreach{d => println(d.user + "===" + d.product + "===" + d.rating)}
    println("==============baixianghui============")
    predictions.foreach{p => println(p.user + "===" + p.product + "===" + p.rating)}
    val s = data.map(x => ((x.user, x.product), x.rating))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(s)
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }
}
