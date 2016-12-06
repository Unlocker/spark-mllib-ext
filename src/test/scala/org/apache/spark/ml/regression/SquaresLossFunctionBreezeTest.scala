package org.apache.spark.ml.regression

import java.lang.System.out

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.optimize.LBFGS
import org.scalatest.{FunSpec, Matchers}

import scala.math.BigDecimal.RoundingMode.HALF_UP

/**
  * Test suite for [[SquaresLossFunctionBreeze]]
  */
class SquaresLossFunctionBreezeTest extends FunSpec with Matchers {

  /**
    * Initial weights
    */
  val initWeights = BDV(1.0, 1.0)
  /**
    * Training data
    */
  val xydata: BDM[Double] = BDM.create(4, 2, Array(0.5, 1, 1.5, 2, 0.5, 1, 4.5, 8))
  /**
    * SUT
    */
  val subj = new SquaresLossFunctionBreeze(new StubNonLinearFunction(), xydata)
  /**
    * Scale
    */
  val scale = 3

  describe("Breeze square loss function") {

    it("should make calculation") {
      val expectedGradient = BDV[Double](-33, -20.285)
      val (loss, grad) = subj.calculate(initWeights)

      loss should equal(22.5) // the half of the sum of partial loss squares
      grad.length should equal(subj.dim)
      grad map { dec => BigDecimal(dec).setScale(scale, HALF_UP).toDouble } should be(expectedGradient)
    }

    it("should calculate the diff vector") {
      val expectedDiff = BDV[Double](0d, 0d, -3d, -6d)
      subj.diff(initWeights) map { dec => BigDecimal(dec).setScale(scale, HALF_UP).toDouble } should be(expectedDiff)
    }

    it("should calculate the Jacobian") {
      val expectedJacobian = BDM((0.5, -0.347), (1.0, 0.0), (1.5, 0.608), (2.0, 1.386))
      val jacobian: BDM[Double] = subj.jacobian(initWeights)

      jacobian map { dec => BigDecimal(dec).setScale(scale, HALF_UP).toDouble } should be(expectedJacobian)
    }

    it("should calculate the Hessian") {
      val expectedHessian = BDM((7.5, 3.512), (3.512, 2.412))
      val hessian: BDM[Double] = subj.hessian(initWeights)

      hessian map { dec => BigDecimal(dec).setScale(scale, HALF_UP).toDouble } should be(expectedHessian)
    }
  }

  describe("Optimization with Breeze loss function") {
    it("should decreaze loss function value") {
      val (initialLoss, _) = subj.calculate(initWeights)

      val optimizer = new LBFGS[BDV[Double]]
      val state: optimizer.State = optimizer.minimizeAndReturnState(subj, initWeights)

      out.println(s"Iter#${state.iter}; Weights=${state.x}; Loss=${state.value}")
      state.value should (be < initialLoss)
    }
  }
}
