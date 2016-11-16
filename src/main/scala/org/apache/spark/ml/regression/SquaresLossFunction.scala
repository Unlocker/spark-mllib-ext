package org.apache.spark.ml.regression

import breeze.linalg.eigSym.DenseEigSym
import breeze.linalg.{eigSym, DenseMatrix => BDM, DenseVector => BDV}
import breeze.optimize.DiffFunction

/**
  * The loss function described as sum of partial loss squares.
  *
  * @see http://www.nodalpoint.com/non-linear-regression-using-spark-part2-sum-of-squares
  */
trait SquaresLossFunction extends DiffFunction[BDV[Double]] {
  /**
    * The model dimensionality (the number of weights).
    *
    * @return dimensionality
    */
  def dim: Int

  /**
    * Evaluates loss function value and the gradient vector
    *
    * @param weights weights
    * @return (loss function value, gradient vector)
    */
  def calculate(weights: BDV[Double]): (Double, BDV[Double])

  /**
    * Calculates a positive definite approximation of the Hessian matrix.
    *
    * @param weights weights
    * @return Hessian matrix approximation
    */
  def hessian(weights: BDV[Double]): BDM[Double]

  /**
    * Enforces positive definiteness of the Hessian matrix.
    *
    * @param H hessian
    * @return positive definitive hessian matrix
    */
  def posDef(H: BDM[Double]): BDM[Double] = {
    val eigens: DenseEigSym = eigSym(H)
    val vectors = eigens.eigenvectors
    val values = eigens.eigenvalues
    val I = BDM.eye[Double](dim)
    for (i <- 0 until dim) {
      I(i, i) = if (values(i) < 1e-4) 1e-4 else values(i)
    }
    vectors * I * vectors.t
  }
}
