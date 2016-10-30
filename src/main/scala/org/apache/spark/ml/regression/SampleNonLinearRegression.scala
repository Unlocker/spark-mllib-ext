/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.regression

import java.lang.Math.exp

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable


trait SampleNonLinearRegressionParams extends PredictorParams
  with HasMaxIter with HasTol with HasWeightCol

/**
  * Sample non-linear regression.
  */
class SampleNonLinearRegression(override val uid: String)
  extends Regressor[Vector, SampleNonLinearRegression, SampleNonLinearRegressionModel]
    with SampleNonLinearRegressionParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("sampleNonLinReg"))

  override def copy(extra: ParamMap): SampleNonLinearRegression = defaultCopy(extra)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  setDefault(maxIter -> 100)

  def setTol(value: Double): this.type = set(tol, value)

  setDefault(tol -> 1e-6)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  override protected def train(dataset: Dataset[_]): SampleNonLinearRegressionModel = {
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)).cast(DoubleType), w, col($(featuresCol))).rdd.map {
      case Row(label: Double, weight: Double, features: Vector) => Instance(label, weight, features)
    }

    // persists dataset if defined any storage level
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val costFunc = new LeastSquaresCostFunction(instances)
    val optimizer = new LBFGS[BDV[Double]]($(maxIter), 10, $(tol))
    // TODO remove the hardcode
    val initialCoef = Vectors.zeros(6)
    val states = optimizer.iterations(new CachedDiffFunction[BDV[Double]](costFunc),
      initialCoef.asBreeze.toDenseVector)
    val (coefficients, objectiveHistory) = {
      val builder = mutable.ArrayBuilder.make[Double]
      var state: optimizer.State = null
      while (states.hasNext) {
        state = states.next()
        builder += state.adjustedValue
      }

      // checks is method failed
      if (state == null) {
        val msg = s"${optimizer.getClass.getName} failed."
        logError(msg)
        throw new SparkException(msg)
      }

      (Vectors.dense(state.x.toArray.clone()).compressed, builder.result)
    }

    if (handlePersistence) instances.unpersist()
    val model = copyValues(new SampleNonLinearRegressionModel(uid, coefficients))
    val (summaryModel: SampleNonLinearRegressionModel, predictionColName: String) =
      model.findSummaryModelAndPredictionCol()

    val trainingSummary = new SampleNonLinearRegressionSummary(summaryModel.transform(dataset),
      predictionColName, $(labelCol), $(featuresCol), model, Array(0D)
    )
    model.setSummary(trainingSummary)
  }
}

object SampleNonLinearRegression
  extends DefaultParamsReadable[SampleNonLinearRegression] {

  /**
    * Calculates function value for the provided coefficients and features.
    * F(x) = a1 + a2*exp(-a3*(x-a0)**2) + a4*atan(a5*(x-a0))
    *
    * @param coef     coefficients
    * @param features features
    * @return function value
    */
  def functionValue(coef: Vector, features: Vector): Double = {
    require(coef.size == 6, s"Expected coefficients size is 6, but provided is ${coef.size}")
    require(features.size == 1, s"Expected features size is 1, but provided is ${features.size}")
    val x = features(0) - coef(0)
    coef(1) + coef(2) * exp(-coef(3) * x * x) + coef(4) * Math.atan(coef(5) * x)
  }

  def firstPartialDerivative(coef: Vector, features: Vector): Array[Double] = {
    require(coef.size == 6, s"Expected coefficients size is 6, but provided is ${coef.size}")
    require(features.size == 1, s"Expected features size is 1, but provided is ${features.size}")

    val x = features(0) - coef(0)
    val exponent: Double = exp(-coef(3) * x * x)
    val divisor: Double = 1 + Math.pow(coef(5) * x, 2)
    // calculated partial derivatives for the objective function
    val pda0 = -2 * coef(2) * coef(3) * x * exponent + coef(4) * coef(5) / divisor
    val pda1 = 1
    val pda2 = exponent
    val pda3 = -coef(2) * x * x * exponent
    val pda4 = Math.atan(coef(5) * x)
    val pda5 = coef(4) * x / divisor
    Array[Double](pda0, pda1, pda2, pda3, pda4, pda5)
  }

  override def load(path: String): SampleNonLinearRegression = super.load(path)
}

/**
  * Model produced by [[SampleNonLinearRegression]]
  *
  * @param uid          UID
  * @param coefficients coefficients
  */
class SampleNonLinearRegressionModel(override val uid: String, val coefficients: Vector)
  extends RegressionModel[Vector, SampleNonLinearRegressionModel]
    with SampleNonLinearRegressionParams
    with MLWritable {

  private var trainingSummary: Option[SampleNonLinearRegressionSummary] = None

  override val numFeatures: Int = coefficients.size

  def setSummary(summary: SampleNonLinearRegressionSummary): this.type = {
    this.trainingSummary = Some(summary)
    this
  }

  def summary: SampleNonLinearRegressionSummary = trainingSummary.getOrElse {
    throw new SparkException("No training summary available for this model")
  }

  def hasSummary: Boolean = trainingSummary.isDefined

  def findSummaryModelAndPredictionCol(): (SampleNonLinearRegressionModel, String) = {
    $(predictionCol) match {
      case "" =>
        val predictionColName = "prediction_" + java.util.UUID.randomUUID.toString
        (copy(ParamMap.empty).setPredictionCol(predictionColName), predictionColName)
      case p => (this, p)
    }
  }

  override def write: MLWriter = new SampleNonLinearRegressionModel.SampleNonLinearModelWriter(this)

  override protected def predict(features: Vector): Double = SampleNonLinearRegression.functionValue(coefficients, features)

  override def copy(extra: ParamMap): SampleNonLinearRegressionModel = {
    val newModel = copyValues(new SampleNonLinearRegressionModel(uid, coefficients), extra)
    if (trainingSummary.isDefined) newModel.setSummary(trainingSummary.get)
    newModel.setParent(parent)
  }
}

object SampleNonLinearRegressionModel extends MLReadable[SampleNonLinearRegressionModel] {

  class SampleNonLinearModelWriter(instance: SampleNonLinearRegressionModel)
    extends MLWriter with Logging {

    private case class Data(coefficients: Vector)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.coefficients)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class SampleNonLinearModelReader extends MLReader[SampleNonLinearRegressionModel] {

    private val className = classOf[SampleNonLinearRegressionModel].getName

    override def load(path: String): SampleNonLinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.format("parquet").load(dataPath)
      val Row(coefficients: Vector) = MLUtils.convertVectorColumnsToML(data, "coefficients").select("coefficients").head()
      val model = new SampleNonLinearRegressionModel(metadata.uid, coefficients)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  override def read: MLReader[SampleNonLinearRegressionModel] = new SampleNonLinearModelReader
}

/**
  * LeastSquaresAggregator computes the gradient and loss for a Least-squared loss function.
  *
  * @param coefficients coefficients vector
  */
private class LeastSquaresAggregator(coefficients: Vector)
  extends Serializable {

  private var totalCnt: Long = 0L
  private var weightSum: Double = 0.0
  private var lossSum = 0.0

  private val dim = coefficients.size
  private val gradientSumArray = Array.ofDim[Double](dim)

  /**
    * Evaluates instance error and updates counters.
    *
    * @param instance instance
    * @return this object
    */
  def add(instance: Instance): this.type = {
    instance match {
      case Instance(label, weight, features) =>
        if (weight == 0.0) return this

        val diff = SampleNonLinearRegression.functionValue(coefficients, features) - label
        if (diff != 0) {
          val localGradientSumArray = gradientSumArray
          val gradientArray = SampleNonLinearRegression.firstPartialDerivative(coefficients, features)
          gradientArray.zipWithIndex.foreach {
            case (x: Double, i: Int) => localGradientSumArray(i) += x * weight
          }
          // squared error accumulator
          lossSum += weight * diff * diff / 2.0
        }
        totalCnt += 1
        weightSum += weight
        this
    }
  }

  /**
    * Merges another LeastSquaresAggregator and updates the loss and gradient of the objective function.
    *
    * @param other object to be merged
    * @return this object
    */
  def merge(other: LeastSquaresAggregator): this.type = {
    require(dim == other.dim, s"Dimensions mismatch when merging with another " +
      s"LeastSquaresAggregator. Expecting $dim but got ${other.dim}.")

    if (other.weightSum != 0) {
      totalCnt += other.totalCnt
      weightSum += other.weightSum
      lossSum += other.lossSum

      val localThisGradientSumArray = this.gradientSumArray
      val localOtherGradientSumArray = other.gradientSumArray
      for (i <- 0 until dim) {
        localThisGradientSumArray(i) += localOtherGradientSumArray(i)
      }
    }
    this
  }

  /**
    * The number of processed instances.
    */
  def count: Long = totalCnt

  /**
    * Accumulated loss
    */
  def loss: Double = {
    lossSum / weightSum
  }

  /**
    * Accumulated gradient
    */
  def gradient: Vector = {
    val result = Vectors.dense(gradientSumArray.clone())
    BLAS.scal(1.0 / weightSum, result)
    result
  }
}

/**
  * LeastSquaresCostFun implements Breeze's DiffFunction[T] for Least Squares cost.
  * It's used in Breeze's convex optimization routines.
  */
private class LeastSquaresCostFunction(instances: RDD[Instance]) extends DiffFunction[BDV[Double]] {
  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val coeffs = Vectors.fromBreeze(coefficients)

    val aggregator = {
      val mapOperator = (c: LeastSquaresAggregator, instance: Instance) => c.add(instance)
      val reduceOperator = (c1: LeastSquaresAggregator, c2: LeastSquaresAggregator) => c1.merge(c2)
      instances.treeAggregate(new LeastSquaresAggregator(coeffs))(mapOperator, reduceOperator)
    }

    val totalGradient = aggregator.gradient.toArray
    (aggregator.loss, new BDV[Double](totalGradient))
  }
}

class SampleNonLinearRegressionSummary(@transient val predictions: DataFrame,
                                       val predictionCol: String,
                                       val labelCol: String,
                                       val featuresCol: String,
                                       private val privateModel: SampleNonLinearRegressionModel,
                                       private val diagInvAtWA: Array[Double]) extends Serializable {

  @transient private val metrics = new RegressionMetrics(
    predictions.select(col(predictionCol), col(labelCol).cast(DoubleType)).rdd.map {
      case Row(pred: Double, label: Double) => (pred, label)
    }, false
  )

  val explainedVariance: Double = metrics.explainedVariance

  val meanAbsoluteError: Double = metrics.meanAbsoluteError

  val meanSquaredError: Double = metrics.meanSquaredError

  val rootMeanSquaredError: Double = metrics.rootMeanSquaredError

  val r2: Double = metrics.r2

}
