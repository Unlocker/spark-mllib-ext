package org.apache.spark.ml.regression

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.Instance
import org.apache.spark.rdd.RDD

/**
  * Spark RDD implementation for the squares loss function.
  *
  * @param fitmodel concrete model implementation
  * @param xydata   RDD with instances
  */
class SquaresLossFunctionRdd(val fitmodel: NonlinearFunction, val xydata: RDD[Instance])
  extends SquaresLossFunction {

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
    val bcW: Broadcast[BDV[Double]] = xydata.context.broadcast(weights)

    val (f: Double, grad: BDV[Double]) = xydata.treeAggregate((0.0, BDV.zeros[Double](dim)))(
      seqOp = (comb, item) => (comb, item) match {
        case ((loss, oldGrad), Instance(label, _, features)) =>
          val featuresBdv = features.asBreeze.toDenseVector
          val w: BDV[Double] = bcW.value
          val prediction = fitmodel.eval(w, featuresBdv)
          val addedLoss: Double = 0.5 * math.pow(label - prediction, 2)
          val addedGrad: BDV[Double] = 2.0 * (prediction - label) * fitmodel.grad(w, featuresBdv)
          (loss + addedLoss, oldGrad + addedGrad)
      },
      combOp = (comb1, comb2) => (comb1, comb2) match {
        case ((loss1, grad1: BDV[Double]), (loss2, grad2: BDV[Double])) => (loss1 + loss2, grad1 + grad2)
      })

    (f, grad)
  }

  /**
    * Calculates a positive definite approximation of the Hessian matrix.
    *
    * @param weights weights
    * @return Hessian matrix approximation
    */
  override def hessian(weights: BDV[Double]): BDM[Double] = {
    val bcW = xydata.context.broadcast(weights)

    val (hessian: BDM[Double]) = xydata.treeAggregate(new BDM[Double](dim, dim, Array.ofDim[Double](dim * dim)))(
      seqOp = (comb, item) => (comb, item) match {
        case ((oldHessian), Instance(_, _, features)) =>
          val grad = fitmodel.grad(bcW.value, features.asBreeze.toDenseVector)
          val subHessian: BDM[Double] = grad * grad.t
          oldHessian + subHessian
      },
      combOp = (comb1, comb2) => (comb1, comb2) match {
        case (hessian1, hessian2) => hessian1 + hessian2
      }
    )

    posDef(hessian)
  }
}
