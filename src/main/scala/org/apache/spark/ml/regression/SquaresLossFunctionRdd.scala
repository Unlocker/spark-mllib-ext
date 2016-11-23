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

    val (grad: BDV[Double], f: Double) = xydata.map(inst => (inst.label, inst.features))
      .treeAggregate((BDV.zeros(dim), 0.0))(
        seqOp = (c, v) => (c, v) match {
          case ((oldGrad, loss), (label, features)) =>
            val feat: BDV[Double] = BDV(features.toArray)
            val w: BDV[Double] = BDV(bcW.value.toArray)
            val pred = fitmodel.eval(w, feat)
            val gradPred = fitmodel.grad(w, feat)
            val addedLoss: Double = 0.5 * math.pow(label - pred, 2)
            val addedGrad: BDV[Double] = 2.0 * (pred - label) * gradPred
            (oldGrad + addedGrad, loss + addedLoss)
        },
        combOp = (c1, c2) => (c1, c2) match {
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
  override def hessian(weights: BDV[Double]): BDM[Double] = ???
}
