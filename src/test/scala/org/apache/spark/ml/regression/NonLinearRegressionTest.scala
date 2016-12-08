package org.apache.spark.ml.regression

import breeze.linalg.DenseVector
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.types.DataTypes._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SQLImplicits}

/**
  * Test suite for [[NonLinearRegression]].
  * It uses function f(x)=a1+a2*exp(-a3*(x-a0)**2)+a4*atan(a5*(x-a0))
  */
class NonLinearRegressionTest extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  /**
    * Testing stuff to define DataFrame implicit conversions
    */
  private object testImplicits extends SQLImplicits {
    protected override def _sqlContext: SQLContext = spark.sqlContext
  }

  import testImplicits._

  /**
    * Function f(x)=a1+a2*exp(-a3*(x-a0)**2)+a4*atan(a5*(x-a0))
    */
  object SampleNonLinearFunction extends StubNonLinearFunction {

    override def eval(w: DenseVector[Double], x: DenseVector[Double]): Double = {
      val y = x(0) - w(0)
      w(1) + w(2) * math.exp(-w(3) * math.pow(y, 2)) + w(4) * math.atan(w(5) * y)
    }

    override def dim: Int = 6
  }

  @transient var dataset: DataFrame = _

  @transient var subject: NonLinearRegression = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    // reads data from CSV
    val x = StructField("x", DoubleType)
    val labelStrict = StructField("label_strict", DoubleType)
    val labelNoised = StructField("label_noised", DoubleType)
    val schema = StructType(Array(x, labelStrict, labelNoised))

    dataset = spark.sqlContext
      .read
      .schema(schema)
      .option("header", value = true)
      .csv(getClass.getResource("/noised_nonlinear_function.csv").getFile)

    subject = new NonLinearRegression(SampleNonLinearFunction)
    subject.setInitCoeffs(Array[Double](22, 2, 2, 2, 2, 2))
  }

  /**
    * Strict labels produced as function value calculation with weights=(25, 1, 1, 1, 1, 1)
    */
  test("Model with strict labels should be fitted") {
    assertModelCreation("label_strict", 0.01)
  }

  /**
    * Noised labels produced from the strict labels by adding random noise uniformly distributed in [-0.5, 0.5]
    */
  test("Model with noised labels should be fitted") {
    assertModelCreation("label_noised", 0.05)
  }

  /**
    * Asserts model retrieval from the regressor.
    *
    * @param labelColumn  column name with label
    * @param mseThreshold MSE threshold
    */
  def assertModelCreation(labelColumn: String, mseThreshold: Double): Unit = {
    val examples = dataset
      .map { (row: Row) => Instance(row.getAs[Double](labelColumn), 1.0, Vectors.dense(row.getAs[Double]("x"))) }
      .toDF()

    val model = subject.fit(examples)

    logWarning("COEFFS=[" + model.coefficients.toArray.mkString(",")
      + s"]; EXPLAINED_VARIANCE=${model.summary.explainedVariance}; MSE=${model.summary.meanSquaredError}"
    )

    assert(model.hasSummary === true)
    assert(model.summary.meanSquaredError < mseThreshold)
  }
}
