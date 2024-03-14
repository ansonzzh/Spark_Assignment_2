import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object Spark2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local[*]").appName("Titanic Survival Prediction").getOrCreate()

    val trainData = spark.read.option("inferSchema", "true").option("header", "true").csv("src/titanic/train.csv")
    val testData = spark.read.option("inferSchema", "true").option("header", "true").csv("src/titanic/test.csv")

    val filledTrainData = fillMissingValues(trainData)
    val filledTestData = fillMissingValues(testData)

    val Array(trainingData, testDataSplit) = filledTrainData.randomSplit(Array(0.8, 0.2))

    val labelIndexer = new StringIndexer().setInputCol("Survived").setOutputCol("indexedLabel").fit(trainingData)
    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("genderIndex")
    val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("embarkedIndex")
    val assembler = new VectorAssembler().setInputCols(Array("Pclass", "genderIndex", "Age", "SibSp", "Parch", "Fare", "embarkedIndex")).setOutputCol("features")

    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("features").setNumTrees(10)
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, genderIndexer, embarkedIndexer, assembler, rf, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testDataSplit)
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy = $accuracy")

    val testPredictions = model.transform(filledTestData)
    testPredictions.select("PassengerId", "predictedLabel").show()

    spark.stop()
  }

  def fillMissingValues(dataFrame: DataFrame): DataFrame = {
    val ageMedian = dataFrame.stat.approxQuantile("Age", Array(0.5), 0.0).head
    val fareMedian = dataFrame.stat.approxQuantile("Fare", Array(0.5), 0.0).head
    val mostCommonEmbarked = dataFrame.groupBy("Embarked").count().orderBy(desc("count")).first().getString(0)

    dataFrame.na.fill(Map(
      "Age" -> ageMedian,
      "Fare" -> fareMedian,
      "Embarked" -> mostCommonEmbarked
    ))
  }
}
