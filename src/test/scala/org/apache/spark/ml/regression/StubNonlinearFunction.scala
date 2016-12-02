package org.apache.spark.ml.regression

import breeze.linalg.DenseVector

/**
  * Implements sample function with two features.
  * {{{
  *   f(x) = b_1*x^{b_2}
  * }}}
  */
class StubNonlinearFunction extends NonlinearFunction with Serializable {

  /**
    * Step to calculate the gradient.
    */
  val step: Double = 1e-6

  /**
    * Evaluates function value.
    *
    * @param weights weights
    * @param x       instance features
    * @return function value
    */
  override def eval(weights: DenseVector[Double], x: DenseVector[Double]): Double = {
    val Array(a, b) = weights.data
    a * math.pow(x(0), b)
  }

  /**
    * Evaluates gradient vector.
    *
    * @param weights weights
    * @param x       instance features
    * @return gradient vector
    */
  override def grad(weights: DenseVector[Double], x: DenseVector[Double]): DenseVector[Double] = {
    // evaluates the gradient numerically
    val initial = eval(weights, x)
    val numGrads = (0 until dim) map (i => {
      val dw: DenseVector[Double] = weights.copy
      dw(i) += step
      dw
    }) map (dw => (eval(dw, x) - initial) / step)
    DenseVector(numGrads.toArray)
  }

  /**
    * The model dimensionality (the number of weights).
    *
    * @return dimensionality
    */
  override def dim: Int = 2
}
