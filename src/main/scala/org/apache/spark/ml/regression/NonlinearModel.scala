package org.apache.spark.ml.regression

import breeze.linalg.{DenseVector => BDV}

/**
  * The generic definition of non-linear model.
  *
  * @see http://www.nodalpoint.com/nonlinear-regression-using-spark-part-1-nonlinear-models/
  */
trait NonlinearModel {
  /**
    * Evaluates function value.
    *
    * @param weights weights
    * @param x       instance features
    * @return function value
    */
  def eval(weights: BDV[Double], x: BDV[Double]): Double

  /**
    * Evaluates gradient vector.
    *
    * @param weights weights
    * @param x       instance features
    * @return gradient vector
    */
  def grad(weights: BDV[Double], x: BDV[Double]): BDV[Double]

  /**
    * The model dimensionality (the number of weights).
    *
    * @return dimensionality
    */
  def dim: Int
}
