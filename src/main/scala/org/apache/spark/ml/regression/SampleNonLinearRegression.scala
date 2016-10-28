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
import breeze.optimize.DiffFunction
import org.apache.spark.internal.Logging
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset


trait SampleNonLinearRegressionParams extends PredictorParams
  with HasMaxIter with HasTol with HasFitIntercept with HasStandardization
  with HasWeightCol with HasSolver

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

  override protected def train(dataset: Dataset[_]): SampleNonLinearRegressionModel = {
    throw new UnsupportedOperationException
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
    require(coef.size == 1, s"Expected features size is 1, but provided is ${features.size}")
    val x = features(0) - coef(0)
    coef(1) + coef(2) * exp(-coef(3) * x * x) + coef(4) * Math.atan(coef(5) * x)
  }

  def firstPartialDerivative(coef: Vector, features: Vector): Array[Double] = {
    require(coef.size == 6, s"Expected coefficients size is 6, but provided is ${coef.size}")
    require(coef.size == 1, s"Expected features size is 1, but provided is ${features.size}")

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

  override def write: MLWriter = {
    throw new UnsupportedOperationException
  }

  override protected def predict(features: Vector): Double = {
    throw new UnsupportedOperationException
  }

  override def copy(extra: ParamMap): SampleNonLinearRegressionModel = {
    throw new UnsupportedOperationException
  }

  private class LeastSquaresCostFun(instances: RDD[Instance])
    extends DiffFunction[BDV[Double]] {

    override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
      throw new UnsupportedOperationException
    }

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

        var i = 0
        val localThisGradientSumArray = this.gradientSumArray
        val localOtherGradientSumArray = other.gradientSumArray
        while (i < dim) {
          localThisGradientSumArray(i) += localOtherGradientSumArray(i)
          i += 1
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

}
