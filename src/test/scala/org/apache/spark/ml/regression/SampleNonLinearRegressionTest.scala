package org.apache.spark.ml.regression

import org.apache.spark.ml.linalg.Vectors
import org.scalatest.{FlatSpec, Matchers}

/**
  * Test suite for [[SampleNonLinearRegression]]
  */
class SampleNonLinearRegressionTest extends FlatSpec with Matchers {

  "Function value" should "calculate as intended" in {
    val features = Vectors.dense(Array[Double](25))
    val coef = Vectors.dense(Array[Double](25, 1, 1, 1, 1, 1))
    val result = SampleNonLinearRegression.functionValue(coef, features)

    result should equal (2.0)
  }

  "First derivative function value" should "calculate as intended" in {
    val features = Vectors.dense(Array[Double](25))
    val coef = Vectors.dense(Array[Double](25, 1, 1, 1, 1, 1))
    val result = SampleNonLinearRegression.firstPartialDerivative(coef, features)

    result should equal (Array[Double](-1, 1, 1, 0, 0, 0))
  }
}
