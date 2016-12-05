package org.apache.spark.ml.regression

import breeze.linalg.{DenseVector => BreezeVector}

/**
  * The generic definition of non-linear function used to learn the model.
  *
  * @see http://www.nodalpoint.com/nonlinear-regression-using-spark-part-1-nonlinear-models/
  */
trait NonlinearFunction extends Serializable {
  /**
    * Evaluates function value.
    *
    * @param weights weights
    * @param x       instance features
    * @return function value
    */
  def eval(weights: BreezeVector[Double], x: BreezeVector[Double]): Double

  /**
    * Evaluates gradient vector.
    *
    * @param weights weights
    * @param x       instance features
    * @return gradient vector
    */
  def grad(weights: BreezeVector[Double], x: BreezeVector[Double]): BreezeVector[Double]

  /**
    * The model dimensionality (the number of weights).
    *
    * @return dimensionality
    */
  def dim: Int
}
