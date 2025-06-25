import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, count, countDistinct, date_format, desc, explode, format_number, row_number, split, sum, to_date, when, avg}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Main {
  val RESET = "\u001b[0m"
  val GREEN = "\u001b[32m"
  val BLUE = "\u001b[34m"
  val CYAN = "\u001b[36m"
  val YELLOW = "\u001b[33m"
  val RED = "\u001b[31m"

  def main(args: Array[String]): Unit = {
    // Configuration de Log4j
    println(getClass.getResource("/log4j.properties"))
    val spark = SparkSession.builder().appName("Diabetes Analysis").master("local[*]").getOrCreate()

    val filePath = getClass.getResource("/diabetes_prediction_dataset.csv").getPath
    val df: DataFrame = spark.read.option("header", "true").option("inferSchema", "true").csv(filePath)
    

    println(s"${GREEN}==== Aperçu des 10 premières lignes du jeu de données ====${RESET}")
    df.show(10, truncate=false)

    // Afficher le schéma
    println(s"${CYAN}==== Schéma des colonnes ====${RESET}")
    df.printSchema()

    // Nombre de lignes
    println(s"${YELLOW}Nombre de lignes dans le dataset : ${df.count()}${RESET}")

    // Valeurs manquantes par colonne
    println(s"${BLUE}==== Valeurs manquantes par colonne ====${RESET}")
    val missing = df.select(df.columns.map(c => sum(when(col(c).isNull || col(c) === "", 1).otherwise(0)).alias(c)): _*)
    missing.show(false)


    // Détection des valeurs aberrantes (exemples)
    // Age
    println(s"${RED}Ages aberrants :${RESET}")
    df.filter(col("age") < 0 || col("age") > 120).show(truncate=false)

    // BMI
    println(s"${RED}BMI aberrants :${RESET}")
    df.filter(col("bmi") < 10 || col("bmi") > 60).show(truncate=false)

    // HbA1c_level
    println(s"${RED}HbA1c_level aberrants :${RESET}")
    df.filter(col("HbA1c_level") < 3 || col("HbA1c_level") > 15).show(truncate=false)

    // blood_glucose_level
    println(s"${RED}Blood Glucose aberrants :${RESET}")
    df.filter(col("blood_glucose_level") < 40 || col("blood_glucose_level") > 300).show(truncate=false)

    // Aperçu random de 5 lignes
    println(s"${CYAN}Sample aléatoire :${RESET}")
    df.sample(0.01).show(5, truncate=false)
  

    // Nombre de patients par classe de diabète
    println(s"${YELLOW}==== Nombre de patients par classe de diabète ====${RESET}")
    df.groupBy("diabetes").count().show()

    // Distribution des classes de diabète
    println(s"${BLUE}==== Distribution des classes de diabète ====${RESET}")
    df.groupBy("diabetes").count().orderBy(desc("count")).show()

    // Nombre de patients par classe de diabète et sexe
    println(s"${YELLOW}==== Nombre de patients par classe de diabète et sexe ====${RESET}")
    df.groupBy("diabetes").count()
      .withColumnRenamed("count", "Nombre de patients")
      .orderBy(desc("Nombre de patients"))
      .show()

    // PARTIE 1: Nettoyage avancé 
    println(s"${BLUE}PARTIE 1: Nettoyage avancé des données${RESET}")

    // Détection des doublons
    val duplicateCount = df.count() - df.dropDuplicates().count()
    println(s"${YELLOW}Nombre de doublons : $duplicateCount${RESET}")

    // Suppression des doublons
    val cleanedData = df.dropDuplicates()
    println(s"${GREEN}Nombre d'entrées après suppression des doublons : ${cleanedData.count()}${RESET}")

    // Statistiques descriptives
    println(s"${CYAN}Statistiques descriptives après nettoyage :${RESET}")
    cleanedData.describe().show(false)

    // PARTIE 2: Analyse avancée 
    println(s"${BLUE}PARTIE 2: Analyse avancée des données${RESET}")

    // Nombre de cas de diabète
    println(s"${GREEN}Nombre de cas de diabète :${RESET}")
    cleanedData.groupBy("diabetes")
      .agg(count("*").alias("count"))
      .withColumn("percentage", format_number(col("count") / cleanedData.count() * 100, 2))
      .show()

    // Analyse par genre
    println(s"${CYAN}Prévalence du diabète par genre :${RESET}")
    cleanedData.groupBy("gender", "diabetes")
      .agg(count("*").alias("count"))
      .orderBy("gender", "diabetes")
      .show()

    // Analyse par âge (groupes d'âge)
    println(s"${BLUE}Prévalence du diabète par groupe d'âge :${RESET}")
    val ageGroups = cleanedData.withColumn("age_group",
      when(col("age") < 30, "<30")
        .when(col("age") >= 30 && col("age") < 45, "30-44")
        .when(col("age") >= 45 && col("age") < 60, "45-59")
        .otherwise("60+")
    )
    ageGroups.groupBy("age_group", "diabetes")
      .agg(count("*").alias("count"))
      .orderBy("age_group", "diabetes")
      .show()

    // Moyennes des indicateurs selon le statut diabétique
    println(s"${YELLOW}Moyenne des indicateurs par statut diabétique :${RESET}")
    cleanedData.groupBy("diabetes")
      .agg(
        avg("age").alias("avg_age"),
        avg("bmi").alias("avg_bmi"),
        avg("HbA1c_level").alias("avg_HbA1c"),
        avg("blood_glucose_level").alias("avg_glucose")
      )
      .show()

    // Analyse des facteurs de risque (hypertension, heart_disease)
    println(s"${RED}Analyse des facteurs de risque (hypertension et maladie cardiaque) :${RESET}")
    cleanedData.groupBy("hypertension", "heart_disease", "diabetes")
      .agg(count("*").alias("count"))
      .orderBy("hypertension", "heart_disease", "diabetes")
      .show()

    println(s"${CYAN}Distribution des modalités de smoking_history :${RESET}")
    df.groupBy("smoking_history").count().orderBy(desc("count")).show(false)

    val dfFixed = df.withColumn(
      "smoking_history",
      when(col("smoking_history") === "ever", "never")
        .otherwise(col("smoking_history"))
    )

    println(s"${CYAN}Distribution des modalités de smoking_history apres traitement :${RESET}")
    dfFixed.groupBy("smoking_history").count().orderBy(desc("count")).show(false)

     val dfWithSmokingNum = dfFixed.withColumn("smoking_info_num",
      when(col("smoking_history") === "never", 0)
        .when(col("smoking_history") === "not current", 1)
        .when(col("smoking_history") === "former", 1)
        .when(col("smoking_history") === "current", 2)
        .when(col("smoking_history") === "no info", -1)
        .otherwise(-1)
    )

      dfWithSmokingNum.groupBy("smoking_history", "smoking_info_num").count().show()

  val fixedData = dfWithSmokingNum
  .withColumn("gender_num",
    when(col("gender") === "Female", 0)
      .when(col("gender") === "Male", 1)
      .otherwise(-1)
  )

  
  val finalData = fixedData.select(
    col("diabetes"),
    col("age"),
    col("hypertension"),
    col("heart_disease"),
    col("bmi"),
    col("HbA1c_level"),
    col("blood_glucose_level"),
    col("gender_num").alias("gender"),
    col("smoking_info_num").alias("smoking_history")
  )
  .filter(col("gender") =!= -1 && col("smoking_history") =!= -1)
  .coalesce(1)


  finalData
  .write
  .option("header", "true")
  .mode("overwrite")
  .csv("./diabetes_cleaned")

  // On enlève le genre et smoking_history pour la deuxième version
  val finalData2 = fixedData
    .select(
      col("diabetes"),
      col("age"),
      col("hypertension"),
      col("heart_disease"),
      col("bmi"),
      col("HbA1c_level"),
      col("blood_glucose_level")
    )
    .coalesce(1)
  
  finalData2
    .write
    .option("header", "true")
    .mode("overwrite")
    .csv("./diabetes_cleaned2")
}
}