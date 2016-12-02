package org.apache.spark.ml.regression

import breeze.linalg.DenseVector
import org.scalatest.{FunSpec, Matchers}

import scala.math.BigDecimal.RoundingMode.HALF_UP

/**
  * Test suite.
  */
class StubNonlinearFunctionTest extends FunSpec with Matchers {

  val func: NonlinearFunction = new StubNonlinearFunction()
  val weights = DenseVector(Array(2.0, 2.0))
  val x: DenseVector[Double] = DenseVector[Double](Array(1.5))

  describe("Non linear function f(x) = b_1*x^{b_2}") {
    it("has correct dimensionality") {
      func.dim should equal(2)
    }

    it(s"has function value for x=$x and b=$weights") {
      func.eval(weights, x) should equal(4.5)
    }

    it(s"has gradient value for x=$x and b=$weights") {
      val grad: DenseVector[Double] = func.grad(weights, x)
      val expectations = DenseVector(2.250, 1.825)

      grad should not be null
      grad.length should equal(func.dim)

      grad map { i => BigDecimal(i).setScale(3, HALF_UP).toDouble } should be(expectations)
    }
  }
}
