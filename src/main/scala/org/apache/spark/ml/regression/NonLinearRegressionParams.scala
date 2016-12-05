package org.apache.spark.ml.regression

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol, HasWeightCol}

/**
  * Non-linear regression parameters.
  */
private[regression] trait NonLinearRegressionParams
  extends PredictorParams with HasMaxIter with HasTol with HasWeightCol
