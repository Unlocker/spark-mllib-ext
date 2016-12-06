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

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, LBFGS}
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Column, Dataset, Row}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

/**
  * Non-linear regression.
  *
  * @param uid    unique identifier
  * @param kernel a non linear regression function
  */
class NonLinearRegression(override val uid: String, val kernel: NonLinearFunction)
  extends Regressor[Vector, NonLinearRegression, NonLinearRegressionModel]
    with NonLinearRegressionParams
    with DefaultParamsWritable
    with Logging {

  /**
    * Non-linear regression.
    *
    * @param kernel a non linear regression function
    * @return non-linear regression
    */
  def this(kernel: NonLinearFunction) = this(Identifiable.randomUID("nonLinReg"), kernel)

  def setInitCoeffs(value: Array[Double]): this.type = set(initCoeffs, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  setDefault(maxIter -> 100)

  def setTol(value: Double): this.type = set(tol, value)

  setDefault(tol -> 1e-6)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  override def copy(extra: ParamMap): NonLinearRegression = defaultCopy(extra)

  /**
    * Trains the regressor to produce model.
    *
    * @param dataset training set
    * @return model
    */
  override protected def train(dataset: Dataset[_]): NonLinearRegressionModel = {
    // set instance weight to 1 if not defined the column
    val instanceWeightCol: Column = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)).cast(DoubleType), instanceWeightCol, col($(featuresCol))).rdd.map {
      case Row(label: Double, weight: Double, features: Vector) => Instance(label, weight, features)
    }

    // persists dataset if defined any storage level
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val costFunc = new SquaresLossFunctionRdd(kernel, instances)
    val optimizer = new LBFGS[BDV[Double]]($(maxIter), 10, $(tol))

    // checks and assigns the initial coefficients
    val initial = {
      if (!isDefined(initCoeffs) || $(initCoeffs).length != kernel.dim) {
        if ($(initCoeffs).length != kernel.dim)
          logWarning(s"Provided initial coefficients vector (${$(initCoeffs)}) not corresponds with model dimensionality equals to ${kernel.dim}")
        BDV.zeros[Double](kernel.dim)
      } else
        BDV($(initCoeffs).clone())
    }

    val states = optimizer.iterations(new CachedDiffFunction[BDV[Double]](costFunc), initial)
    val (coefficients, objectiveHistory) = {
      val builder = mutable.ArrayBuilder.make[Double]
      var state: optimizer.State = null
      while (states.hasNext) {
        state = states.next
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
    // unpersists the instances RDD
    if (handlePersistence) instances.unpersist()

    val model = copyValues(new NonLinearRegressionModel(uid, kernel, coefficients))
    val (summaryModel: NonLinearRegressionModel, predictionColName: String) = model.findSummaryModelAndPredictionCol()
    val trainingSummary = new NonLinearRegressionSummary(
      summaryModel.transform(dataset), predictionColName, $(labelCol), $(featuresCol), objectiveHistory, model
    )
    model.setSummary(trainingSummary)
  }
}

/**
  * Non-linear regression companion object.
  */
object NonLinearRegression extends DefaultParamsReadable[NonLinearRegression] {

  override def load(path: String): NonLinearRegression = super.load(path)
}
