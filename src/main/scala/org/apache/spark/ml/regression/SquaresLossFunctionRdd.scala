package org.apache.spark.ml.regression

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.rdd.RDD

/**
  * Spark RDD implementation for the squares loss function.
  *
  * @param fitmodel concrete model implementation
  * @param xydata   RDD with instances
  */
class SquaresLossFunctionRdd(val fitmodel: NonlinearModel, val xydata: RDD[Instance])
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
    val bcW = xydata.context.broadcast(weights)

    val (grad: BDV[Double], f: Double) = xydata.map(inst => (inst.label, BDV(inst.features.toArray)))
      .treeAggregate((BDV.zeros(dim), 0.0))(
        seqOp = (combiner, item) => (combiner, item) match {
          case ((oldGrad, loss), (label, features)) =>
            val w: BDV[Double] = BDV(bcW.value.toArray)
            val prediction = fitmodel.eval(w, features)
            val addedLoss: Double = 0.5 * math.pow(label - prediction, 2)
            val addedGrad: BDV[Double] = 2.0 * (prediction - label) * fitmodel.grad(w, features)
            (oldGrad + addedGrad, loss + addedLoss)
        },
        combOp = (combiner1, combiner2) => (combiner1, combiner2) match {
          case ((grad1: BDV[Double], loss1), (grad2: BDV[Double], loss2)) => (grad1 + grad2, loss1 + loss2)
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

    val (hessian: BDM[Double]) = xydata.map(inst => (inst.label, BDV(inst.features.toArray)))
      .treeAggregate(new BDM[Double](dim, dim, new Array[Double](dim * dim)))(
        seqOp = (combiner, item) => (combiner, item) match {
          case ((oldHessian), (label, features)) =>
            val w: BDV[Double] = BDV(bcW.value.toArray)
            val grad = fitmodel.grad(w, features)
            val subHessian: BDM[Double] = grad * grad.t
            oldHessian + subHessian
        },
        combOp = (combiner1, combiner2) => (combiner1, combiner2) match {
          case (hessian1, hessian2) => hessian1 + hessian2
        }
      )

    posDef(hessian)
  }
}
