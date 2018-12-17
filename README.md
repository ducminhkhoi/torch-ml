# Implementation of some fundamental Machine Learning algorithms completely written in Pytorch

[PyTorch](https://pytorch.org/) is a framework that consists of:

* Tensor library for GPU and CPU
* Autograd (Automatic Differentiation) and Optimization library (SGD, Adam, ...)
* Neural Network library (CNN, RNN, FC)

**Requirements**:

* Pytorch 1.0:
* Sklearn (for getting data)

The pseudocodes of the algorithms are taken from the book: [Machine Learning: An Algorithmic Perspective, Second Edition](https://www.amazon.com/Machine-Learning-Algorithmic-Perspective-Recognition/dp/1466583282/ref=sr_1_1?ie=UTF8&qid=1545027451&sr=8-1&keywords=machine+learning+an+algorithmic+perspective)

**Main idea**: Instead of relying other tools for solving convex optimization problems, I use SGD to solve them directly.

In sum: all machine learning algorithms that can be defined in term of loss (function of parameters needed to estimate) and that loss is differentiable can be solved by using Pytorch.

I already implemented some fundamental Machine Learning algorithms:

* [Chapter 3: Perceptron](chapter3.py)
* [Chapter 4: Multilayer Perceptron](chapter4.py)
* [Chapter 5: Radial Basis Function](chapter5.py)
* [Chapter 6: Dimension Reduction: Linear Discriminant Analysis, Principal Component Analysis, Local Linear Embedding, Autoencoder](chapter6.py)
* [Chapter 7: Clustering](chapter7.py)
* [Chapter 8: Support Vector Machines](chapter8.py)
* [Chapter 13: Ensemble Learning](chapter13.py)
* [Chapter 18: Gaussian Processes](chapter18.py)
