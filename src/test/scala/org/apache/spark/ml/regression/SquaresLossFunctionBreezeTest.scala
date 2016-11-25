package org.apache.spark.ml.regression

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.scalatest.{FunSpec, Matchers}

/**
  * Test suite for [[SquaresLossFunctionBreeze]]
  */
class SquaresLossFunctionBreezeTest extends FunSpec with Matchers {

  val initWeights = BDV(1.0, 1.0)
  val xydata: BDM[Double] = BDM.create(4, 2, Array(0.5, 0.5, 1, 1, 1.5, 4.5, 2, 8))
  val subj = new SquaresLossFunctionBreeze(new StubNonlinearFunction(), xydata)

  describe("Breeze square loss function") {
    it("should make calculation") {
      System.out.println(subj.calculate(initWeights))
    }
  }
}
