# 03.Forecasting-automobile-pricing
Using Machine learning algorithms and stastical tools, we made a prediction of automobile pricing using a database of 13 million records. We used the following methodology 

1. Preprocessing: 

  1.1 We used some techniques to reduce the number of variables (columns) and records (rows) for this problem, we used correlation coefficient, entropy       measure, Kolmogorov-Smirnov test, non parametric test (Kruskal Wallis), outlier detection such that we improved quality.
  of data for training the machine learning models. 

2. Feature engineering:

  2.1 In this step, we propose three different methodologies for data. We created a low-dimensional database using PCA and SVD, and a high-dimensional       database using radial basis kernels. The aim of these databases is to compare machine learning models with different input sizes.
  
  
3. Training Machine Learning Models

  3.1 First, we use PAC and VC-dimension to know how many data points are needed as a minimum for some learning machines. 
 
  3.2 We implemented the following supervised algorithms: Linear regression, KNN, Neural networks and Decision trees.
 
  3.3 Using grid search, we fixed the hyperparameters of the models.
  
Finally, we obtained the best results with the high-dimensional database using radial basis kernel and neural networks
