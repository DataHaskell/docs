---
layout: page
title: "Current Environment"
category: community
date: 2016-10-19 21:08:18
---

A list of data science and machine learning tools and algorithms that either already exist or we would like to exist.

Contents :

- [Data structures](#datastructures)
  - [Data frames](#dataframes)
  - [Efficient arrays](#arrays)
  
- [Numerical methods](#numerical)
  - [Numerical linear algebra](#nla)
  - [Integration](#integration)
    - Markov chain Monte Carlo
  - [Differentiation](#differentiation)
    - Automatic differentiation
  - [Optimization](#optimization)   
    - Linear programming
    - Convex optimization
  
- [Machine learning](#ml)
  - [Bayesian inference](#bayes)
  - [Graphical models](#graph)
  - [Classification](#classification)   
    - Neural networks
    - Naive Bayes
    - Kernel methods
    - Ensemble methods
  - [Regression](#regression) 
    - Nearest neighbors
    - Linear regression
    - Gaussian processes
    - Kalman filtering
  - [Reinforcement learning](#rl)     
  - [Dimensionality reduction](#dimr)    


<h2 id="datastructures">Data structures</h2>


<h3 id="dataframes">Data frames</h3>

- **frames** - [hackage](http://hackage.haskell.org/package/Frames) : User-friendly, type safe, runtime efficient tooling for working with tabular data deserialized from comma-separated values (CSV) files. The type of each row of data is inferred from data, which can then be streamed from disk, or worked with in memory. Also see the comprehensive [tutorial](https://acowley.github.io/Frames/)
- **labels** - [hackage](https://hackage.haskell.org/package/labels) : Declare and access tuple fields with labels. An approach to anonymous records.
- **analyze** - [hackage](https://hackage.haskell.org/package/analyze), [github](https://github.com/ejconlon/analyze) : `pandas`-like dataframe operations for tabular data with CSV interface.
- **kraps-h** - [github](https://github.com/krapsh/kraps-haskell) : Haskell bindings to Apache Spark. The library consists of: 
  * a specification to describe data pipelines in a language-agnostic manner, and a communication protocol to submit these pipelines to Spark. 
  * a serving library, called krapsh-server, that implements this specification on top of Spark. It is written in Scala and is loaded as a standard Spark package.
  * a client written in Haskell that sends pipelines to Spark for execution. In addition, this client serves as an experimental platform for whole-program optimization and verification, as well as compiler-enforced type checking.

<h3 id="arrays">Efficient arrays</h3>

- **vector** - [hackage](https://hackage.haskell.org/package/vector) : An efficient implementation of Int-indexed arrays (both mutable and immutable), with a powerful loop optimisation framework.
- **accelerate** - [hackage](https://hackage.haskell.org/package/accelerate) : Data.Array.Accelerate defines an embedded array language for computations for high-performance computing in Haskell. Computations on multi-dimensional, regular arrays are expressed in the form of parameterised collective operations, such as maps, reductions, and permutations. These computations may then be online compiled and executed on a range of architectures.
- **repa** - [hackage](https://hackage.haskell.org/package/repa) : Repa provides high performance, regular, multi-dimensional, shape polymorphic parallel arrays. All numeric data is stored unboxed. 


<h2 id="numerical">Numerical methods</h2>

<h3 id="nla">Numerical linear algebra</h3>

- **hmatrix** - [hackage](http://hackage.haskell.org/package/hmatrix) : Bindings to BLAS/LAPACK. Linear solvers, matrix decompositions, and more.
- **sparse-linear-algebra** - [hackage](https://hackage.haskell.org/package/sparse-linear-algebra) : Native library for sparse algebraic computation. Linear solvers, matrix decompositions and related tools; functional but not optimized for efficiency yet.


<h3 id="integration">Integration</h3>

<h4 id="mcmc">Markov Chain Monte Carlo</h4>

  - **declarative** - [hackage](https://hackage.haskell.org/package/declarative) : A simple combinator language for Markov transition operators that are useful in MCMC.
  - **mwc-probability**  -  [hackage](https://hackage.haskell.org/package/mwc-probability) : A simple probability distribution type, where distributions are characterized by sampling functions.



<h3 id="differentiation">Differentiation</h3>

<h4 id="ad">Automatic differentiation</h4>

  - **ad** - [hackage](http://hackage.haskell.org/package/ad) : Automatic differentiation to arbitrary order, applicable to data provided in any Traversable container.


<h3 id="optimization">Optimization</h3>

<h4 id="lp">Linear programming</h4>

  - **glpk-hs** - [hackage](https://hackage.haskell.org/package/glpk-hs) : Friendly interface to GLPK's linear programming and mixed integer programming features. Intended for easy extensibility, with a general, pure-Haskell representation of linear programs. 


<h4 id="cvxopt">Convex optimization</h4>
  - **optimization** - [hackage](https://hackage.haskell.org/package/optimization) : A number of optimization techniques from the modern optimization literature (quasi-Newton, stochastic gradient descent, mirror descent, projected subgradient etc.).



<h2 id="ml">Machine learning</h2>


<h3 id="bayes">Bayesian inference</h3>
  - **probably-baysig** - [github](https://github.com/glutamate/probably-baysig) - This library contains definitions and functions for probabilistic and statistical inference.
    * Math.Probably.Sampler defines the sampling function monad, as described by Sungwoo Park and implemented elsewhere (e.g. 'random-fu' and 'monte-carlo' packages)
    * Math.Probably.PDF defines some common parametric log-probability density functions
    * Math.Probably.FoldingStats defines statistics as folds that can be composed and calculated independently of the container of the underlying data.
    * Strategy.* implements various transition operators for Markov Chain Monte Carlo, including Metropolis-Hastings, Hamiltonian Monte Carlo, NUTS, and continuous/discrete slice samplers.
    * Math.Probably.MCMC implements functions and combinators for running Markov chains and interleaving transition operators.


<h3 id="supervised">Supervised learning</h3>


<h4 id="graph">Graphical models</h4>

<h5 id="hmm">Hidden Markov models</h5>
  - **HMM** - [hackage](http://hackage.haskell.org/package/HMM), [github](https://github.com/mikeizbicki/hmm)
  - **hmm-hmatrix** - [hackage](http://hackage.haskell.org/package/hmm-hmatrix), [darcs](http://hub.darcs.net/thielema/hmm-hmatrix)
  - **learning-hmm** - [hackage](http://hackage.haskell.org/package/learning-hmm), [github](https://github.com/mnacamura/learning-hmm)


<h4 id="classification">Classification</h4>

- Neural Networks
  - Simple Neural Networks
    - **sibe** - [hackage](http://hackage.haskell.org/package/sibe), [github](https://github.com/mdibaiee/sibe)
    - **neural** - [hackage](http://hackage.haskell.org/package/neural), [github](https://github.com/brunjlar/neural)
  - Recurrent Neural Networks
  - Convolutional Neural Networks
    - **tensorflow** - [github](https://github.com/tensorflow/haskell) : Haskell bindings for TensorFlow
  - Generating Neural Networks
 
 References : [NNL-Speak for Haskellers](https://colah.github.io/posts/2015-09-NN-Types-FP/)



- Naive Bayes
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
       - **sibe** - [hackage](http://hackage.haskell.org/package/sibe), [github](https://github.com/mdibaiee/sibe)
  - Bernoulli Naive Bayes

- Kernel methods
  - Support Vector Machines
    - with custom kernel function (beside Gaussian, Chi², etc.)

  - Core Vector Machines
    - including approximative Center-Constrained-Minimum-Enclosing-Ball via Core-Set
    - Classification & Regression
    - O(n), but with huge constants

- Ensemble methods
  - Decision Trees
      - Random Forests
  - Boosting 
      - XGBoost 
        - **xgboost.hs** [github](https://github.com/robertzk/xgboost.hs)
      - AdaBoost    

<h4 id="regression">Regression</h4>

- Nearest Neighbors
  - **HLearn** - [Blogpost](https://izbicki.me/blog/fast-nearest-neighbor-queries-in-haskell.html) (including github-links)

- Linear Regression
  - **statistics** - [hackage](http://hackage.haskell.org/package/statistics), [github](https://github.com/bos/statistics)
  
- Gaussian processes
  - **HasGP** - [hackage](https://hackage.haskell.org/package/HasGP) : Gaussian processes for regression and classification, based on the Laplace approximation and Expectation Propagation.
  
- Kalman filtering
  - **kalman** - [hackage](https://hackage.haskell.org/package/kalman) : Linear, extended and unscented Kalman filters are provided, along with their corresponding smoothers. Furthermore, a particle filter and smoother is provided.
  - **estimator** - [hackage](https://hackage.haskell.org/package/estimator) : The goal of this library is to simplify implementation and use of state-space estimation algorithms, such as Kalman Filters. The interface for constructing models is isolated as much as possible from the specifics of a given algorithm, so swapping out a Kalman Filter for a Bayesian Particle Filter should involve a minimum of effort.
This implementation is designed to support symbolic types, such as from sbv or ivory. As a result you can generate code in another language, such as C, from a model written using this package; or run static analyses on your model.
  

<h4 id="rl">Reinforcement learning</h4>

- Policy gradient
- Q-Learning
     - Neural Network Q-Learning

<h4 id="clustering">Clustering</h4>

- K-Means
  - **kmeans** - [hackage](https://hackage.haskell.org/package/kmeans), [darcs](http://hub.darcs.net/gershomb/kmeans)
- Self-Organising Maps (SOM)
  - Hyperbolic-SOM
  - Hierarchical (H)SOMs
- Mean-shift
- Affinity propagation
- Spectral Clustering
- Ward hierarchical clustering
- Birch

<h4 id="dimr">Dimensionality reduction</h4>

- Principal Component Analysis (PCA)
  - **sibe** - [hackage](http://hackage.haskell.org/package/sibe), [github](https://github.com/mdibaiee/sibe)
  - Kernel PCA
  - Incremental PCA
  - Truncated SVD
  
- Independent Component Analysis (ICA)

- t-SNE (t-distributed stochastic neighbor embedding)


Contribute
====

If you know a library that has to do with Data Science, please consider [adding](https://github.com/DataHaskell/docs/edit/gh-pages/_posts/2016-10-19-wishlist.md) it, if the category it belongs to doesn't exist, suggest a category for it.

Add sections related to data science and not only Machine Learning such as Data Mining, Distributed Processing, etc
