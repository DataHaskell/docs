---
layout: page
title: "Current Environment"
category: community
date: 2016-10-19 21:08:18
---

A list of data science and machine learning algorithms we would like to have,
along with existing implementations (if any).

Numerical methods
-----------------

### Numerical linear algebra

- hmatrix - [hackage](http://hackage.haskell.org/package/hmatrix) : Bindings to BLAS/LAPACK. Linear solvers, matrix decompositions, and more.
- sparse-linear-algebra - [hackage](https://hackage.haskell.org/package/sparse-linear-algebra) : Native library for sparse algebraic computation. Linear solvers, matrix decompositions and related tools; functional but not optimized for efficiency yet.

### Integration

- Markov chain Monte Carlo
  - declarative - [hackage](https://hackage.haskell.org/package/declarative) : A simple combinator language for Markov transition operators that are useful in MCMC.
  - mwc-probability  -  [hackage](https://hackage.haskell.org/package/mwc-probability) : A simple probability distribution type, where distributions are characterized by sampling functions.


### Differentiation

- Automatic differentiation
  - ad - [hackage](http://hackage.haskell.org/package/ad) : Automatic differentiation to arbitrary order, applicable to data provided in any Traversable container.


### Optimization

- Linear programming
  - glpk-hs - [hackage](https://hackage.haskell.org/package/glpk-hs) : Friendly interface to GLPK's linear programming and mixed integer programming features. Intended for easy extensibility, with a general, pure-Haskell representation of linear programs. 

- Convex optimization
  - optimization - [hackage](https://hackage.haskell.org/package/optimization) : A number of optimization techniques from the modern optimization literature (quasi-Newton, stochastic gradient descent, mirror descent, projected subgradient etc.).


Machine Learning
----------------

### Graphical Models

- Hidden Markov Models
  - HMM - [hackage](http://hackage.haskell.org/package/HMM), [github](https://github.com/mikeizbicki/hmm)
  - hmm-hmatrix - [hackage](http://hackage.haskell.org/package/hmm-hmatrix), [darcs](http://hub.darcs.net/thielema/hmm-hmatrix)
  - learning-hmm - [hackage](http://hackage.haskell.org/package/learning-hmm), [github](https://github.com/mnacamura/learning-hmm)

### Supervised Learning

#### Classification

- Neural Networks
  - Simple Neural Networks
    - sibe - [hackage](http://hackage.haskell.org/package/sibe), [github](https://github.com/mdibaiee/sibe)
    - neural - [hackage](http://hackage.haskell.org/package/neural), [github](https://github.com/brunjlar/neural)
    - grenade - [hackage](http://hackage.haskell.org/package/grenade), [github](https://github.com/HuwCampbell/grenade)
  - Recurrent Neural Networks
    - grenade (not released to Hackage yet) - [github](https://github.com/HuwCampbell/grenade)
  - Convolutional Neural Networks
    - grenade - [hackage](http://hackage.haskell.org/package/grenade), [github](https://github.com/HuwCampbell/grenade)
  - LTSM (Long Term Short term Memory)
    - grenade - [hackage](http://hackage.haskell.org/package/grenade), [github](https://github.com/HuwCampbell/grenade)
  - Generating Neural Networks
  - Basically everything of: [NNL-Speak for Haskellers](https://colah.github.io/posts/2015-09-NN-Types-FP/)


- Nearest Neighbors
  - HLearn: [Blogpost](https://izbicki.me/blog/fast-nearest-neighbor-queries-in-haskell.html) (including github-links)

- Naive Bayes
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
       - sibe - [hackage](http://hackage.haskell.org/package/sibe), [github](https://github.com/mdibaiee/sibe)
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
  - AdaBoost    

#### Regression

- Linear Regression
  - statistics - [hackage](http://hackage.haskell.org/package/statistics), [github](https://github.com/bos/statistics)
  
- Gaussian processes
  - HasGP - [hackage](https://hackage.haskell.org/package/HasGP) : Gaussian processes for regression and classification, based on the Laplace approximation and Expectation Propagation.
  

### Reinforcement Learning
- Policy gradient
- Q-Learning
     - Neural Network Q-Learning

### Clustering
- K-Means
  - kmeans - [hackage](https://hackage.haskell.org/package/kmeans), [darcs](http://hub.darcs.net/gershomb/kmeans)
- Self-Organising Maps (SOM)
  - Hyperbolic-SOM
  - Hierarchical (H)SOMs
- Mean-shift
- Affinity propagation
- Spectral Clustering
- Ward hierarchical clustering
- Birch


### Dimensionality Reduction
- Principal Component Analysis (PCA)
  - sibe - [hackage](http://hackage.haskell.org/package/sibe), [github](https://github.com/mdibaiee/sibe)
  - Kernel PCA
  - Incremental PCA
  - Truncated SVD

- Independent Component Analysis (ICA)

- t-SNE (t-distributed stochastic neighbor embedding)

Contribute
====

If you know a library that has to do with Data Science, please consider [adding](https://github.com/DataHaskell/docs/edit/gh-pages/_posts/2016-10-19-wishlist.md) it, if the category it belongs to doesn't exist, suggest a category for it.

Add sections related to data science and not only Machine Learning such as Data Mining, Distributed Processing, etc
