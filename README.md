**Machine Learning extension library for Apache Spark**

Nonlinear models often used to approximate input data in the signal processing. 
These models usually have shorter weights vector length than typical linear one but demand to calculate the Jacobian and the Hessian matrices. 
Using the Newton-Gauss method there are some difficulties to calculate loss function gradient and the inverted Hessian in distributed context.

Authors in the article (http://ieeexplore.ieee.org/document/5451114/) propose incremental calculation method. 
It uses the Jacobian matrix row by row without full matrix calculation. 

In the blog notes authored by Constantinos Voglis proposed the Spark application of method mentioned.
* http://www.nodalpoint.com/nonlinear-regression-using-spark-part-1-nonlinear-models
* http://www.nodalpoint.com/non-linear-regression-using-spark-part2-sum-of-squares

I am trying to adapt code from blog with Apache Spark 2.0 and to refine Scala implementation.
  