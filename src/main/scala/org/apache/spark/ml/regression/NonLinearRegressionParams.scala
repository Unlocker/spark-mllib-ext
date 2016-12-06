package org.apache.spark.ml.regression

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol, HasWeightCol}
import org.apache.spark.ml.param.{DoubleArrayParam, Params}

private[regression] trait HasInitCoeffs extends Params {
  /**
    * Param to specify the initial coefficients vector to start learning from.
    *
    * @group param
    */
  final val initCoeffs: DoubleArrayParam = new DoubleArrayParam(
    this, "initCoeff", "specifies the initial coefficients vector to start learning from"
  )

  /** @group getParam */
  final def getInitCoeffs: Array[Double] = $(initCoeffs)
}

/**
  * Non-linear regression parameters.
  */
private[regression] trait NonLinearRegressionParams
  extends PredictorParams with HasMaxIter with HasTol with HasWeightCol with HasInitCoeffs
