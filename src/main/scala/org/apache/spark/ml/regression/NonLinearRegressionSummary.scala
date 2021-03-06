package org.apache.spark.ml.regression

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Row}

/**
  * Training summary.
  *
  * @param predictions     DataFrame with predictions
  * @param predictionCol   prediction column name
  * @param labelCol        label column name
  * @param featuresCol     features column name
  * @param learningHistory loss function values through the learning
  * @param privateModel    model
  */
class NonLinearRegressionSummary(@transient val predictions: DataFrame,
                                 val predictionCol: String,
                                 val labelCol: String,
                                 val featuresCol: String,
                                 val learningHistory: Array[Double],
                                 private val privateModel: NonLinearRegressionModel) extends Serializable {
  /**
    * Metrics calculator.
    */
  @transient private val metrics = new RegressionMetrics(
    predictions.select(col(predictionCol), col(labelCol).cast(DoubleType)).rdd.map {
      case Row(pred: Double, label: Double) => (pred, label)
    }, false
  )

  /**
    * Explained variance,
    */
  def explainedVariance: Double = metrics.explainedVariance

  /**
    * Mean absolute error.
    */
  def meanAbsoluteError: Double = metrics.meanAbsoluteError

  /**
    * Mean squared error.
    */
  def meanSquaredError: Double = metrics.meanSquaredError

  /**
    * Root mean squared error.
    */
  def rootMeanSquaredError: Double = metrics.rootMeanSquaredError

  /**
    * Unadjusted coefficient of determination.
    */
  def r2: Double = metrics.r2

}
