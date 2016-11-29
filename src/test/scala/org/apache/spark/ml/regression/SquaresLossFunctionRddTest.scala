package org.apache.spark.ml.regression

import java.lang.System.out

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.optimize.LBFGS
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.scalatest.{BeforeAndAfterEach, FunSpec, Matchers}

import scala.math.BigDecimal.RoundingMode.HALF_UP

/**
  *
  */
class SquaresLossFunctionRddTest extends FunSpec
  with BeforeAndAfterEach
  with Matchers
  with MLlibTestSparkContext {

  @transient var dataset: RDD[Instance] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    // reads data from CSV
    val x = StructField("x", DataTypes.DoubleType)
    val f = StructField("f", DataTypes.DoubleType)
    val schema = StructType(Array(x, f))

    dataset = spark.sqlContext
      .read
      .schema(schema)
      .option("header", value = true)
      .csv(getClass.getResource("/stub_nonlinear_function.csv").getFile)
      .rdd
      .map { row => Instance(row.getDouble(1), 1.0, Vectors.dense(Array[Double](row.getDouble(0)))) }
  }

  override protected def beforeEach(): Unit = {
    super.beforeEach()
    subj = new SquaresLossFunctionRdd(new StubNonlinearFunction(), dataset)
  }

  /**
    * Initial weights
    */
  val initWeights = BDV(1.0, 1.0)
  /**
    * SUT
    */
  @transient var subj: SquaresLossFunctionRdd = _
  /**
    * Scale
    */
  val scale = 3

  describe("RDD square loss function") {

    it("should make calculation") {
      val expectedGradient = BDV[Double](-33, -20.285)
      val (loss, grad) = subj.calculate(initWeights)

      loss should equal(22.5) // the half of the sum of partial loss squares
      grad.length should equal(subj.dim)
      grad map { dec => BigDecimal(dec).setScale(scale, HALF_UP).toDouble } should be(expectedGradient)
    }

    it("should calculate the Hessian") {
      val expectedHessian = BDM((7.5, 3.512), (3.512, 2.412))
      val hessian: BDM[Double] = subj.hessian(initWeights)

      hessian map { dec => BigDecimal(dec).setScale(scale, HALF_UP).toDouble } should be(expectedHessian)
    }
  }

  describe("Optimization with RDD loss function") {
    it("should decreaze loss function value") {
      val (initialLoss, _) = subj.calculate(initWeights)

      val optimizer = new LBFGS[BDV[Double]]
      val state: optimizer.State = optimizer.minimizeAndReturnState(subj, initWeights)

      out.println(s"Iter#${state.iter}; Weights=${state.x}; Loss=${state.value}")
      state.value should (be < initialLoss)
    }
  }
}
