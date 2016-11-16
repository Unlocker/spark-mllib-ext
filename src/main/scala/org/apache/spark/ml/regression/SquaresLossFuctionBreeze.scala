package org.apache.spark.ml.regression

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

/**
  * Breeze implementation for the squares loss function.
  *
  * @param fitmodel concrete model implementation
  * @param xydata   labeled data combined into the matrix:
  *                 the first n-th columns consist of feature values, (n+1)-th columns contains labels
  */
class SquaresLossFuctionBreeze(val fitmodel: NonlinearModel, xydata: BDM[Double])
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
  override def calculate(weights: BDV[Double]): (Double, BDV[Double]) = ???

  /**
    * Calculates the Jacobian matrix
    *
    * @param weights weights
    * @return
    */
  def janal(weights: BDV[Double]): BDM[Double] = {

    val gradData: Array[Double] = (0 until instanceCount) map (i =>
      fitmodel.grad(weights, X(i, ::).t)
      ) flatMap { v => v.data } toArray

    BDM.create(instanceCount, featureCount, gradData)
  }

  /**
    * Calculates the Hessian approximation
    *
    * @param weights weights
    * @return Hessian
    */
  def hanal(weights: BDV[Double]): BDM[Double] = {
    val J: BDM[Double] = janal(weights)
    J.t * J
  }

  /**
    * Calculates a positive definite approximation of the Hessian matrix.
    *
    * @param weights weights
    * @return Hessian matrix approximation
    */
  override def hessian(weights: BDV[Double]): BDM[Double] = posDef(hanal(weights))
}

