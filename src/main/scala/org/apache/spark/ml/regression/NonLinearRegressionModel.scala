package org.apache.spark.ml.regression

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}
import java.util.Base64

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row

/**
  * Model produced by [[NonLinearRegression]]
  *
  * @param uid          UID
  * @param kernel       regression function
  * @param coefficients coefficients
  */
class NonLinearRegressionModel(override val uid: String, val kernel: NonlinearFunction, val coefficients: Vector)
  extends RegressionModel[Vector, NonLinearRegressionModel]
    with NonLinearRegressionParams
    with MLWritable {

  private var trainingSummary: Option[NonLinearRegressionSummary] = None

  override val numFeatures: Int = kernel.dim

  def setSummary(summary: NonLinearRegressionSummary): this.type = {
    this.trainingSummary = Some(summary)
    this
  }

  def summary: NonLinearRegressionSummary = trainingSummary.getOrElse {
    throw new SparkException("No training summary available for this model")
  }

  def hasSummary: Boolean = trainingSummary.isDefined

  def findSummaryModelAndPredictionCol(): (NonLinearRegressionModel, String) = {
    $(predictionCol) match {
      case "" =>
        val predictionColName = "prediction_" + java.util.UUID.randomUUID.toString
        (copy(ParamMap.empty).setPredictionCol(predictionColName), predictionColName)
      case p => (this, p)
    }
  }

  override def write: MLWriter = new NonLinearRegressionModel.NonLinearModelWriter(this)

  override protected def predict(features: Vector): Double = {
    kernel.eval(coefficients.asBreeze.toDenseVector, features.asBreeze.toDenseVector)
  }

  override def copy(extra: ParamMap): NonLinearRegressionModel = {
    val newModel = copyValues(new NonLinearRegressionModel(uid, kernel, coefficients), extra)
    if (trainingSummary.isDefined) newModel.setSummary(trainingSummary.get)
    newModel.setParent(parent)
  }
}

/**
  * Utility object for the [[NonLinearRegressionModel]].
  */
object NonLinearRegressionModel extends MLReadable[NonLinearRegressionModel] {

  /**
    * Converts serializable object into Base64 string.
    *
    * @param item object to convert
    * @tparam T serializable type
    * @return base64 string
    */
  def serialize[T <: scala.Serializable](item: T): String = {
    val baos = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(baos)
    oos.writeObject(item)
    oos.close()
    Base64.getEncoder.encodeToString(baos.toByteArray)
  }

  /**
    * Converts Base64 string into serializable object.
    *
    * @param data base64 string
    * @tparam T serializable type
    * @return object
    */
  def deserialize[T <: scala.Serializable](data: String): T = {
    val bytes: Array[Byte] = Base64.getDecoder.decode(data)
    val ois = new ObjectInputStream(new ByteArrayInputStream(bytes))
    val obj = ois.readObject()
    ois.close()
    obj.asInstanceOf[T]
  }

  /**
    * Model writer.
    *
    * @param instance model
    */
  private class NonLinearModelWriter(instance: NonLinearRegressionModel) extends MLWriter with Logging {

    private case class Data(kernelData: String, coefficients: Vector)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(serialize(instance.kernel), instance.coefficients)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  /**
    * Model reader.
    */
  private class NonLinearModelReader extends MLReader[NonLinearRegressionModel] {

    private val className = classOf[NonLinearRegressionModel].getName

    override def load(path: String): NonLinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.format("parquet").load(dataPath)
      val Row(kernelData: String, coefficients: Vector) = MLUtils
        .convertVectorColumnsToML(data, "coefficients").select("kernelData", "coefficients").head()

      val model = new NonLinearRegressionModel(metadata.uid, deserialize(kernelData), coefficients)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  override def read: MLReader[NonLinearRegressionModel] = new NonLinearModelReader
}