package org.apache.spark.ml.regression

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

/**
  * Breeze implementation for the squares loss function.
  *
  * @param fitmodel concrete model implementation
  * @param xydata   labeled data combined into the matrix:
  *                 the first n-th columns consist of feature values, (n+1)-th columns contains labels
  */
class SquaresLossFunctionBreeze(val fitmodel: NonlinearFunction, xydata: BDM[Double])
  extends SquaresLossFunction {

  /**
    * The number of instances.
    */
  val instanceCount: Int = xydata.rows
  /**
    * The number of features.
    */
  val featureCount: Int = xydata.cols - 1
  /**
    * Feature matrix.
    */
  val X: BDM[Double] = xydata(::, 0 until featureCount)
  /**
    * Label vector.
    */
  val y: BDV[Double] = xydata(::, featureCount)

  /**
    * The model dimensionality (the number of weights).
    *
    * @return dimensionality
    */
  override def dim: Int = fitmodel.dim

  /**
    * Evaluates loss function value and the gradient vector
    *
    * @param weights weights
    * @return (loss function value, gradient vector)
    */
  override def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
    val r: BDV[Double] = diff(weights)
    (0.5 * (r.t * r), gradient(weights))
  }

  /**
    * Calculates a positive definite approximation of the Hessian matrix.
    *
    * @param weights weights
    * @return Hessian matrix approximation
    */
  override def hessian(weights: BDV[Double]): BDM[Double] = {
    val J: BDM[Double] = jacobian(weights)
    posDef(J.t * J)
  }

  /**
    * Calculates the Jacobian matrix
    *
    * @param weights weights
    * @return the Jacobian
    */
  def jacobian(weights: BDV[Double]): BDM[Double] = {
    val gradData = (0 until instanceCount) map { i => fitmodel.grad(weights, X(i, ::).t).toArray }
    BDM(gradData: _*)
  }

  /**
    * Calculates the difference vector between the label and the approximated values.
    *
    * @param weights weights
    * @return difference vector
    */
  def diff(weights: BDV[Double]): BDV[Double] = {
    val diff = (0 until instanceCount) map (i => fitmodel.eval(weights, X(i, ::).t) - y(i))
    BDV(diff.toArray)
  }

  /**
    * Calculates the gradient vector
    *
    * @param weights weights
    * @return gradient vector
    */
  def gradient(weights: BDV[Double]): BDV[Double] = {
    val J: BDM[Double] = jacobian(weights)
    val r = diff(weights)
    2.0 * J.t * r
  }
}

