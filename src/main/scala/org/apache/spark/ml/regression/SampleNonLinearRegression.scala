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

import org.apache.spark.internal.Logging
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWritable, MLWriter}
import org.apache.spark.sql.Dataset
import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.DiffFunction
import org.apache.spark.ml.feature.Instance
import org.apache.spark.rdd.RDD


trait SampleNonLinearRegressionParams extends PredictorParams
  with HasMaxIter with HasTol with HasFitIntercept with HasStandardization
  with HasWeightCol with HasSolver

/**
  * Sample non-linear regression.
  */
class SampleNonLinearRegression(override val uid: String) extends
  Regressor[Vector, SampleNonLinearRegression, SampleNonLinearRegressionModel]
  with SampleNonLinearRegressionParams with DefaultParamsWritable with Logging {

  override def copy(extra: ParamMap): SampleNonLinearRegression = {
    throw new UnsupportedOperationException
  }

  override protected def train(dataset: Dataset[_]): SampleNonLinearRegressionModel = {
    throw new UnsupportedOperationException
  }
}

class SampleNonLinearRegressionModel(override val uid: String, val coefficients: Vector) extends
  RegressionModel[Vector, SampleNonLinearRegressionModel] with SampleNonLinearRegressionParams with MLWritable {

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

  private class LeastSquaresAggregator(coefficients: Vector)
    extends Serializable {

    private var totalCnt: Long = 0L
    private var weightSum: Double = 0.0
    private var lossSum = 0.0

    private val dim = coefficients.size
    private val gradientSumArray = Array.ofDim[Double](dim)

    /**
      *
      * @param instance
      * @return
      */
    def add(instance: Instance): this.type = {
      instance match {
        case Instance(label, weight, features) =>
          if (weight == 0.0) return this

          // TODO replace 1 with function value
          val diff = 1 - label
          if (diff != 0) {
            val localGradientSumArray = gradientSumArray
            features.foreachActive { (index, value) =>
              if (value != 0.0) {
                localGradientSumArray(index) += weight * diff * value
              }
            }
            lossSum += weight * diff * diff / 2.0
          }
          totalCnt += 1
          weightSum += weight
          this
      }
    }

    def merge(other: LeastSquaresAggregator): this.type = {
      throw new UnsupportedOperationException
    }

    def count: Long = totalCnt

    def loss: Double = {
      lossSum / weightSum
    }

    def gradient: Vector = {
      val result = Vectors.dense(gradientSumArray.clone())
      BLAS.scal(1.0 / weightSum, result)
      result
    }
  }

}
