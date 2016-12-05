package org.apache.spark.ml.regression

import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.DataFrame
import org.scalatest.{FlatSpec, Matchers}

/**
  * Test suite for [[NonLinearRegression]]
  */
class SampleNonLinearRegressionTest extends FlatSpec
  with Matchers with MLlibTestSparkContext {

  @transient var dataset: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    dataset = spark.createDataFrame(
      sc.parallelize(Seq(
        Instance(1, 1, Vectors.dense(-0.5291537477)),
        Instance(2, 1, Vectors.dense(-0.5273454314)),
        Instance(3, 1, Vectors.dense(-0.5253730474)),
        Instance(4, 1, Vectors.dense(-0.5232132235)),
        Instance(5, 1, Vectors.dense(-0.5208379311)),
        Instance(6, 1, Vectors.dense(-0.5182132652)),
        Instance(7, 1, Vectors.dense(-0.5152978215)),
        Instance(8, 1, Vectors.dense(-0.5120405041)),
        Instance(9, 1, Vectors.dense(-0.5083775168)),
        Instance(10, 1, Vectors.dense(-0.504228163)),
        Instance(11, 1, Vectors.dense(-0.499488862)),
        Instance(12, 1, Vectors.dense(-0.4940244355)),
        Instance(13, 1, Vectors.dense(-0.4876550949)),
        Instance(14, 1, Vectors.dense(-0.4801364396)),
        Instance(15, 1, Vectors.dense(-0.4711276743)),
        Instance(16, 1, Vectors.dense(-0.4601391056)),
        Instance(17, 1, Vectors.dense(-0.4464413322)),
        Instance(18, 1, Vectors.dense(-0.4288992722)),
        Instance(19, 1, Vectors.dense(-0.4056476494)),
        Instance(20, 1, Vectors.dense(-0.3734007669)),
        Instance(21, 1, Vectors.dense(-0.3258175511)),
        Instance(22, 1, Vectors.dense(-0.2489223626)),
        Instance(23, 1, Vectors.dense(-0.0888330789)),
        Instance(24, 1, Vectors.dense(0.5824812778)),
        Instance(25, 1, Vectors.dense(2)),
        Instance(26, 1, Vectors.dense(2.1532776046)),
        Instance(27, 1, Vectors.dense(2.1254643567)),
        Instance(28, 1, Vectors.dense(2.2491691822)),
        Instance(29, 1, Vectors.dense(2.3258177762)),
        Instance(30, 1, Vectors.dense(2.373400767)),
        Instance(31, 1, Vectors.dense(2.4056476494)),
        Instance(32, 1, Vectors.dense(2.4288992722)),
        Instance(33, 1, Vectors.dense(2.4464413322)),
        Instance(34, 1, Vectors.dense(2.4601391056)),
        Instance(35, 1, Vectors.dense(2.4711276743)),
        Instance(36, 1, Vectors.dense(2.4801364396)),
        Instance(37, 1, Vectors.dense(2.4876550949)),
        Instance(38, 1, Vectors.dense(2.4940244355)),
        Instance(39, 1, Vectors.dense(2.499488862)),
        Instance(40, 1, Vectors.dense(2.504228163)),
        Instance(41, 1, Vectors.dense(2.5083775168)),
        Instance(42, 1, Vectors.dense(2.5120405041)),
        Instance(43, 1, Vectors.dense(2.5152978215)),
        Instance(44, 1, Vectors.dense(2.5182132652)),
        Instance(45, 1, Vectors.dense(2.5208379311)),
        Instance(46, 1, Vectors.dense(2.5232132235)),
        Instance(47, 1, Vectors.dense(2.5253730474)),
        Instance(48, 1, Vectors.dense(2.5273454314)),
        Instance(49, 1, Vectors.dense(2.5291537477)),
        Instance(50, 1, Vectors.dense(2.5308176397))
      ), 2)
    )
  }

//  "Model" should "be fitted" in {
  //    val trainer = new NonLinearRegression()
  //    val model = trainer.fit(dataset)
  //
  //    model.hasSummary should be(true)
  //
  //    println(s"Coefficients = " + model.coefficients.toArray.mkString(","))
  //    println(s"Explained Variance = ${model.summary.explainedVariance}")
  //    println(s"MSE = ${model.summary.meanSquaredError}")
  //  }
}
