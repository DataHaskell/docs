---
layout: page
title: "Current Environment"
category: community
date: 2016-10-19 21:08:18
---

A list of data science and machine learning tools and algorithms that either already exist or we would like to exist.

## Data structures


### Data frames

- **frames** [](http://hackage.haskell.org/package/Frames){:.hackage} : User-friendly, type safe, runtime efficient tooling for working with tabular data deserialized from comma-separated values (CSV) files. The type of each row of data is inferred from data, which can then be streamed from disk, or worked with in memory. Also see the comprehensive [tutorial](https://acowley.github.io/Frames/)

- **labels** [](https://hackage.haskell.org/package/labels){:.hackage} : Declare and access tuple fields with labels. An approach to anonymous records.

- **analyze** [](https://hackage.haskell.org/package/analyze){:.hackage} [](https://github.com/ejconlon/analyze){:.github} : `pandas`-like dataframe operations for tabular data with CSV interface.

- **kraps-h** [](https://github.com/krapsh/kraps-haskell){:.github} : Haskell bindings to Apache Spark. The library consists of: 
  - A specification to describe data pipelines in a language-agnostic manner, and a communication protocol to submit these pipelines to Spark. 
  - A serving library, called krapsh-server, that implements this specification on top of Spark. It is written in Scala and is loaded as a standard Spark package.
  - A client written in Haskell that sends pipelines to Spark for execution. In addition, this client serves as an experimental platform for whole-program optimization and verification, as well as compiler-enforced type checking.

### Efficient arrays

- **vector** [](https://hackage.haskell.org/package/vector){:.hackage} : An efficient implementation of Int-indexed arrays (both mutable and immutable), with a powerful loop optimisation framework.
- **accelerate** [](https://hackage.haskell.org/package/accelerate){:.hackage} : Data.Array.Accelerate defines an embedded array language for computations for high-performance computing in Haskell. Computations on multi-dimensional, regular arrays are expressed in the form of parameterised collective operations, such as maps, reductions, and permutations. These computations may then be online compiled and executed on a range of architectures.
- **repa** [](https://hackage.haskell.org/package/repa){:.hackage} : Repa provides high performance, regular, multi-dimensional, shape polymorphic parallel arrays. All numeric data is stored unboxed. 


## Numerical methods

### Numerical linear algebra

- **hmatrix** [](http://hackage.haskell.org/package/hmatrix){:.hackage} : Bindings to BLAS/LAPACK. Linear solvers, matrix decompositions, and more.
- **sparse-linear-algebra** [](https://hackage.haskell.org/package/sparse-linear-algebra){:.hackage} : Native library for sparse algebraic computation. Linear solvers, matrix decompositions and related tools; functional but not optimized for efficiency yet.




### Integration

- Markov Chain Monte Carlo
  - **declarative** [](https://hackage.haskell.org/package/declarative){:.hackage} : A simple combinator language for Markov transition operators that are useful in MCMC.
  - **mwc-probability**  [](https://hackage.haskell.org/package/mwc-probability){:.hackage} : A simple probability distribution type, where distributions are characterized by sampling functions.

- Quantiles, etc.
  - **tdigest** [](https://hackage.haskell.org/package/tdigest){:.hackage} : A new data structure for accurate on-line accumulation of rank-based statistics such as quantiles and trimmed means.




### Differentiation

- Automatic differentiation
  - **ad** [](http://hackage.haskell.org/package/ad){:.hackage} : Automatic differentiation to arbitrary order, applicable to data provided in any Traversable container.


### Optimization


- Linear programming
  - **glpk-hs** [](https://hackage.haskell.org/package/glpk-hs){:.hackage} : Friendly interface to GLPK's linear programming and mixed integer programming features. Intended for easy extensibility, with a general, pure-Haskell representation of linear programs. 


- Convex optimization
  - **optimization** [](https://hackage.haskell.org/package/optimization){:.hackage} : A number of optimization techniques from the modern optimization literature (quasi-Newton, stochastic gradient descent, mirror descent, projected subgradient etc.).




 

## Machine learning

### Bayesian inference
  - **probably-baysig** [](https://github.com/glutamate/probably-baysig){:.github} - This library contains definitions and functions for probabilistic and statistical inference.
    - Math.Probably.Sampler defines the sampling function monad, as described by Sungwoo Park and implemented elsewhere (e.g. 'random-fu' and 'monte-carlo' packages)
    - Math.Probably.PDF defines some common parametric log-probability density functions
    - Math.Probably.FoldingStats defines statistics as folds that can be composed and calculated independently of the container of the underlying data.
    - Strategy.\* implements various transition operators for Markov Chain Monte Carlo, including Metropolis-Hastings, Hamiltonian Monte Carlo, NUTS, and continuous/discrete slice samplers.
    - Math.Probably.MCMC implements functions and combinators for running Markov chains and interleaving transition operators.


### Supervised learning


- Graphical models
  - Hidden Markov models
    - **HMM** [](http://hackage.haskell.org/package/HMM){:.hackage} [](https://github.com/mikeizbicki/hmm){:.github}
    - **hmm-hmatrix** [](http://hackage.haskell.org/package/hmm-hmatrix){:.hackage} [](http://hub.darcs.net/thielema/hmm-hmatrix){:.darcs}
    - **learning-hmm** [](http://hackage.haskell.org/package/learning-hmm){:.hackage} [](https://github.com/mnacamura/learning-hmm){:.github}


- Classification

- Neural Networks
  - Simple Neural Networks
    - **sibe** [](http://hackage.haskell.org/package/sibe){:.hackage} [](https://github.com/mdibaiee/sibe){:.github}
    - **neural** [](http://hackage.haskell.org/package/neural){:.hackage} [](https://github.com/brunjlar/neural){:.github}

  - Recurrent Neural Networks

  - Convolutional Neural Networks
    - **tensorflow** [](https://github.com/tensorflow/haskell){:.github} : Haskell bindings for TensorFlow
  
  - Generating Neural Networks
 
  - References : [NNL-Speak for Haskellers](https://colah.github.io/posts/2015-09-NN-Types-FP/)



- Naive Bayes
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
    - **sibe** [](http://hackage.haskell.org/package/sibe){:.hackage} [](https://github.com/mdibaiee/sibe){:.github}
  - Bernoulli Naive Bayes

- Kernel methods
  - Support Vector Machines with 
    - custom kernel function (beside Gaussian, Chi², etc.)

  - Core Vector Machines
    - including approximative Center-Constrained-Minimum-Enclosing-Ball via Core-Set
    - Classification & Regression
    - O(n), but with huge constants

- Ensemble methods
  - Decision Trees
    - Random Forests

- Boosting
  - XGBoost 
    - **xgboost.hs** [](https://github.com/robertzk/xgboost.hs){:.github}
  - AdaBoost    

- Regression

  - Nearest Neighbors
    - **HLearn** [](https://izbicki.me/blog/fast-nearest-neighbor-queries-in-haskell.html){:.blogpost} 
  
  - Linear Regression
    - **statistics** [](http://hackage.haskell.org/package/statistics){:.hackage} [](https://github.com/bos/statistics){:.github}
    
  - Gaussian processes
    - **HasGP** [](https://hackage.haskell.org/package/HasGP){:.hackage} : Gaussian processes for regression and classification, based on the Laplace approximation and Expectation Propagation.
    
  - Kalman filtering
    - **kalman** [](https://hackage.haskell.org/package/kalman){:.hackage} : Linear, extended and unscented Kalman filters are provided, along with their corresponding smoothers. Furthermore, a particle filter and smoother is provided.
    - **estimator** [](https://hackage.haskell.org/package/estimator){:.hackage} : The goal of this library is to simplify implementation and use of state-space estimation algorithms, such as Kalman Filters. The interface for constructing models is isolated as much as possible from the specifics of a given algorithm, so swapping out a Kalman Filter for a Bayesian Particle Filter should involve a minimum of effort.
    This implementation is designed to support symbolic types, such as from sbv or ivory. As a result you can generate code in another language, such as C, from a model written using this package; or run static analyses on your model.
  

- Reinforcement learning
  - Policy gradient

  - Q-Learning
    - Neural Network Q-Learning

- Clustering

  - K-Means
    - **kmeans** [](https://hackage.haskell.org/package/kmeans){:.hackage} [](http://hub.darcs.net/gershomb/kmeans){:.darcs}

  - Self-Organising Maps (SOM)
    - Hyperbolic-SOM
    - Hierarchical (H)SOMs

  - Mean-shift
  - Affinity propagation
  - Spectral Clustering
  - Ward hierarchical clustering
  - Birch

- Dimensionality reduction

  - Principal Component Analysis (PCA)
    - **sibe** [](http://hackage.haskell.org/package/sibe){:.hackage} [](https://github.com/mdibaiee/sibe){:.github}
    - Kernel PCA
    - Incremental PCA
  - Truncated SVD*
  - Independent Component Analysis (ICA)
  - t-SNE (t-distributed stochastic neighbor embedding)
    - **tsne** [](https://hackage.haskell.org/package/tsne){:.hackage}

## Datasets

- **datasets** [](https://hackage.haskell.org/package/datasets){:.hackage} [](https://github.com/glutamate/datasets){:.github} - Classical machine learning and statistics datasets from the UCI Machine Learning Repository and other sources.
The datasets package defines two different kinds of datasets: 
  - Small data sets which are directly (or indirectly with file-embed) embedded in the package as pure values and do not require network or IO to download the data set. This includes **Iris**, **Anscombe** and **OldFaithful**
  - Other data sets which need to be fetched over the network and are cached in a local temporary directory.
- **mnist-idx** [](https://hackage.haskell.org/package/mnist-idx ){:.hackage} [](https://github.com/kryoxide/mnist-idx){:.github} - Read and write data in the IDX format used in e.g. the MINST database


# Contribute

If you know a library that has to do with Data Science, please consider [adding](https://github.com/DataHaskell/docs/edit/gh-pages/_posts/2016-10-19-current-environment.md) it, if the category it belongs to doesn't exist, suggest a category for it.

Add sections related to data science and not only Machine Learning such as Data Mining, Distributed Processing, etc
