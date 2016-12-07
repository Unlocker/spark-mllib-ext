package org.apache.spark.ml.regression

import breeze.linalg.DenseVector
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.types.DataTypes._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SQLImplicits}
import org.scalatest.{FlatSpec, Matchers}

/**
  * Test suite for [[NonLinearRegression]].
  * It uses function f(x)=a1+a2*exp(-a3*(x-a0)**2)+a4*atan(a5*(x-a0))
  */
class NonLinearRegressionTest extends FlatSpec with Matchers with MLlibTestSparkContext {

  /**
    * Function f(x)=a1+a2*exp(-a3*(x-a0)**2)+a4*atan(a5*(x-a0))
    */
  class SampleNonLinearFunction extends StubNonLinearFunction {

    override def eval(w: DenseVector[Double], x: DenseVector[Double]): Double = {
      val y = x(0) - w(0)
      w(1) + w(2) * math.exp(-w(3) * math.pow(y, 2)) + w(4) * math.atan(w(5) * y)
    }

    override def dim: Int = 6
  }

  private object testImplicits extends SQLImplicits {
    protected override def _sqlContext: SQLContext = spark.sqlContext
  }

  import testImplicits._

  @transient var dataset: DataFrame = _

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
  }

  "Model" should "be fitted" in {
    val trainer = new NonLinearRegression(new SampleNonLinearFunction)
    trainer.setInitCoeffs(Array[Double](12, 5, 5, 5, 5, 5))
    val examples: DataFrame = dataset
      .map { (row: Row) =>
        Instance(
          label = row.getAs[Double]("label_strict"),
          weight = 1.0,
          features = Vectors.dense(row.getAs[Double]("x")))
      }.toDF()
    val model = trainer.fit(examples)

    model.hasSummary should be(true)

    println(s"Coefficients = " + model.coefficients.toArray.mkString(","))
    println(s"Explained Variance = ${model.summary.explainedVariance}")
    println(s"MSE = ${model.summary.meanSquaredError}")
  }
}
