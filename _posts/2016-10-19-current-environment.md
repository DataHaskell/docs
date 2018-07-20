---
layout: page
title: "Current Environment"
category: community
date: 2016-10-19 21:08:18
---

A list of data science and machine learning tools and algorithms that either already exist or we would like to exist.



## Data structures

### Data frames

- **frames** [](http://hackage.haskell.org/package/Frames){:.hackage} [![Hackage](https://img.shields.io/hackage/v/Frames.svg)](https://hackage.haskell.org/package/Frames) [![Frames](http://stackage.org/package/Frames/badge/lts)](http://stackage.org/lts/package/Frames) [![Frames](http://stackage.org/package/Frames/badge/nightly)](http://stackage.org/nightly/package/Frames) : User-friendly, type safe, runtime efficient tooling for working with tabular data deserialized from comma-separated values (CSV) files. The type of each row of data is inferred from data, which can then be streamed from disk, or worked with in memory. Also see the comprehensive [tutorial](https://acowley.github.io/Frames/)
- **analyze** [](https://hackage.haskell.org/package/analyze){:.hackage} [](https://github.com/ejconlon/analyze){:.github} [![Hackage](https://img.shields.io/hackage/v/analyze.svg)](https://hackage.haskell.org/package/analyze) [![analyze](http://stackage.org/package/analyze/badge/lts)](http://stackage.org/lts/package/analyze) [![analyze](http://stackage.org/package/analyze/badge/nightly)](http://stackage.org/nightly/package/analyze) : `pandas`-like dataframe operations for tabular data with CSV interface.
- **bookkeeper** [](https://hackage.haskell.org/package/bookkeeper){:.hackage} : A new take on datatypes and records using `OverloadedLabels` (which is available since GHC 8). It bears some similarities to Nikita Volkov's `record` library, but requires no Template Haskell.



### Arrays

- **vector** [](https://hackage.haskell.org/package/vector){:.hackage} : An efficient implementation of Int-indexed arrays (both mutable and immutable), with a powerful loop optimisation framework.
#### Multidimensional arrays
- **accelerate** [](https://hackage.haskell.org/package/accelerate){:.hackage} : Data.Array.Accelerate defines an embedded array language for computations for high-performance computing in Haskell. Computations on multi-dimensional, regular arrays are expressed in the form of parameterised collective operations, such as maps, reductions, and permutations. These computations may then be online compiled and executed on a range of architectures.
- **repa** [](https://hackage.haskell.org/package/repa){:.hackage} : Repa provides high performance, regular, multi-dimensional, shape polymorphic parallel arrays. All numeric data is stored unboxed. 
- **massiv** [](https://hackage.haskell.org/package/massiv){:.hackage} : Repa-style high-performance multi-dimentional arrays with nested parallelism and stencil computation capabilities.

### Records

- **labels** [](https://hackage.haskell.org/package/labels){:.hackage} : Declare and access tuple fields with labels. An approach to anonymous records.
- **superrecord** [](http://hackage.haskell.org/package/superrecord){:.hackage} [](https://github.com/agrafix/superrecord){:.github} Supercharged anonymous records. Introductory [blogpost](https://www.athiemann.net/2017/07/02/superrecord.html), with case study using ReaderT.
- **microgroove** [](https://hackage.haskell.org/package/microgroove){:.hackage} [](https://github.com/daig/microgroove){:.github} : Array-backed extensible records, providing fast access and mutation.

### Graphs

  - **alga** [](https://hackage.haskell.org/package/algebraic-graphs){:.hackage}  Alga is a library for algebraic construction and manipulation of graphs in Haskell. See [this paper](https://github.com/snowleopard/alga-paper) for the motivation behind the library, the underlying theory and implementation details.
The top-level module `Algebra.Graph` defines the core data type `Graph`, which is a deep embedding of four graph construction primitives `empty`, `vertex`, `overlay` and `connect`. To represent non-empty graphs, see `Algebra.Graph.NonEmpty`. More conventional graph representations can be found in `Algebra.Graph.AdjacencyMap` and `Algebra.Graph.Relation`.
The type classes defined in `Algebra.Graph.Class` and `Algebra.Graph.HigherKinded.Class` can be used for polymorphic graph construction and manipulation. Also see `Algebra.Graph.Fold` that defines the Boehm-Berarducci encoding of algebraic graphs and provides additional flexibility for polymorphic graph manipulation.
  - **fgl** [](https://hackage.haskell.org/package/fgl){:.hackage}  An inductive representation of manipulating graph data structures. Original website can be found at http://web.engr.oregonstate.edu/~erwig/fgl/haskell.
  
### Trees

  - **tree-traversals** [](https://hackage.haskell.org/package/tree-traversals-0.1.0.0){:.hackage} The tree-traversals package defines in-order, pre-order, post-order, level-order, and reversed level-order traversals for tree-like types, and it also provides newtype wrappers for the various traversals so they may be used with `traverse`.


## Database interfaces

- **beam** [Homepage](http://tathougies.github.io/beam/) Beam is a highly-general library for accessing any kind of database with Haskell. It supports several backends. beam-postgres and beam-sqlite are included in the main beam repository. Others are hosted and maintained independently, such as beam-mysql and beam-firebird. The documentation here shows examples in all known backends.
Beam is highly extensible and other backends can be shipped independently without requiring any changes in the core libraries.For information on creating additional SQL backends, see the manual section for more.
    - Easy schema generation from existing databases
    - A basic migration infrastructure for working with multiple versions of your database schema.
    - Support for most SQL92, SQL99, and SQL2003 features across backends that support them, including aggregations, subqueries, and window functions.
    - A straightforward Haskell-friendly query syntax. You can use Beam's Q monad much like you would interact with the [] monad.
    - No Template Haskell Beam uses the GHC Haskell type system and nothing else. The types have been designed to be easily-inferrable by the compiler, and appropriate functions to refine types have been provided for the where the compiler may need more help.
- **selda** [](https://github.com/valderman/selda){:.github} Selda is a Haskell library for interacting with SQL-based relational databases inspired by LINQ and Opaleye.
    - Monadic interface.
    - Portable: backends for SQLite and PostgreSQL.
    - Generic: easy integration with your existing Haskell types.
    - Creating, dropping and querying tables using type-safe database schemas.
    - Typed query language with products, filtering, joins and aggregation.
    - Inserting, updating and deleting rows from tables.
    - Conditional insert/update.
    - Transactions, uniqueness constraints and foreign keys.
    - Seamless prepared statements.
    - Configurable, automatic, consistent in-process caching of query results.
    - Lightweight and modular: few dependencies, and non-essential features are optional or split into add-on packages.
- **relational-record** [Homepage](http://khibino.github.io/haskell-relational-record) Haskell Relational Record (HRR) is a query generator based on typed relational algebra and correspondence between SQL value lists and Haskell record types, which provide programming interfaces to Relational DataBase Managemsnt Systems (RDBMS).
  - Abstracted - relations are expressed as high level expressions and they are translated into SQL statements. Drivers are provided for DB2, PostgreSQL, SQLite, MySQL, Microsoft SQL Server and OracleSQL.
  - Type safe - SQL statements produced by HRR are guaranteed to be valid if the Haskell code compiles. Even the types of placeholders are propagated.
  - Composable - relations can be composed to build bigger relations.
  - Automatic - SQL schema is obtained from a target DB and Haskell types are automatically generated at compile time.



## Numerical methods

### Numerical linear algebra

- **hmatrix** [](http://hackage.haskell.org/package/hmatrix){:.hackage} : Bindings to BLAS/LAPACK. Linear solvers, matrix decompositions, and more.
- **sparse-linear-algebra** [](https://hackage.haskell.org/package/sparse-linear-algebra){:.hackage} : Native library for sparse algebraic computation. Linear solvers, matrix decompositions and related tools; functional but not optimized for efficiency yet.

### Generation of random data

  - **mwc-probability**  [](https://hackage.haskell.org/package/mwc-probability){:.hackage} : A simple probability distribution type, where distributions are characterized by sampling functions.
  
  
### Statistics

  - **statistics** [](https://hackage.haskell.org/package/statistics){:.hackage} : This library provides a number of common functions and types useful in statistics. We focus on high performance, numerical robustness, and use of good algorithms. Where possible, we provide references to the statistical literature.
The library's facilities can be divided into four broad categories:
    - Working with widely used discrete and continuous probability distributions. (There are dozens of exotic distributions in use; we focus on the most common.)
    - Computing with sample data: quantile estimation, kernel density estimation, histograms, bootstrap methods, significance testing, and regression and autocorrelation analysis.
    - Random variate generation under several different distributions.
    - Common statistical tests for significant differences between samples.
  - **foldl-statistics** [](https://hackage.haskell.org/package/foldl-statistics){:.hackage} [](https://github.com/data61/foldl-statistics){:.github} : A reimplementation of the Statistics.Sample module using the foldl package. The intention of this package is to allow these algorithms to be used on a much broader set of data input types, including lists and streaming libraries such as conduit and pipes, and any other type which is Foldable.
All statistics in this package can be computed with no more than two passes over the data - once to compute the mean and once to compute any statistics which require the mean.
  - **histogram-fill** [](https://hackage.haskell.org/package/histogram-fill){:.hackage} : A convenient way to create and fill histograms. It supports fixed- and variable-size bins, missing data and 2D binning.
  - **uncertain** [](https://hackage.haskell.org/package/uncertain){:.hackage} : Provides tools to manipulate numbers with inherent experimental/measurement uncertainty, and propagates them through functions.
  - **measurable** [](https://github.com/jtobin/measurable){:.github} Construct measures from samples, mass/density functions, or even sampling functions. Construct image measures by fmap-ing measurable functions over them, or create new measures from existing ones by measure convolution and friends provided by a simple Num instance enabled by an Applicative instance. Create measures from graphs of other measures using the Monad instance and do-notation.
Query measures by integrating meaurable functions against them. Extract moments, cumulative density functions, or probabilities.
Caveat: while fun to play with, and rewarding to see how measures fit together, measure operations as nested integrals are exponentially complex. Don't expect them to scale very far!
 


### Integration

- Markov Chain Monte Carlo
  - **declarative** [](https://hackage.haskell.org/package/declarative){:.hackage} : A simple combinator language for Markov transition operators that are useful in MCMC.
  - **flat-mcmc** [](https://hackage.haskell.org/package/flat-mcmc){:.hackage} : flat-mcmc uses an ensemble sampler that is invariant to affine transformations of space. It wanders a target probability distribution's parameter space as if it had been "flattened" or "unstretched" in some sense, allowing many particles to explore it locally and in parallel.
In general this sampler is useful when you want decent performance without dealing with any tuning parameters or local proposal distributions.


- Quantiles, etc.
  - **tdigest** [](https://hackage.haskell.org/package/tdigest){:.hackage} : A new data structure for accurate on-line accumulation of rank-based statistics such as quantiles and trimmed means.
  
- Dynamical systems
  - **numeric-ode** [](http://hackage.haskell.org/package/numeric-ode){:.hackage} [](https://github.com/qnikst/numeric-ode){:.github} Small project for different ODE solvers, in particular symplectic solvers.
This is very experimental and will change. The Störmer-Verlet generates a correct orbit for Jupiter but no guarantees are given for any of the other methods.




### Differentiation

- Automatic differentiation
  - **ad** [](http://hackage.haskell.org/package/ad){:.hackage} : Automatic differentiation to arbitrary order, applicable to data provided in any Traversable container.
  - **backprop** [](http://hackage.haskell.org/package/backprop){:.hackage} [](https://github.com/mstksg/backprop){:.github} Automatic heterogeneous back-propagation. Write your functions to compute your result, and the library will automatically generate functions to compute your gradient. Differs from `ad` by offering full heterogeneity -- each intermediate step and the resulting value can have different types. Mostly intended for usage with gradient descent and other numeric optimization techniques. Introductory blogpost [here](https://blog.jle.im/entry/introducing-the-backprop-library.html).


### Optimization


- Linear programming
  - **glpk-hs** [](https://hackage.haskell.org/package/glpk-hs){:.hackage} : Friendly interface to GLPK's linear programming and mixed integer programming features. Intended for easy extensibility, with a general, pure-Haskell representation of linear programs. 
- Convex optimization
  - **optimization** [](https://hackage.haskell.org/package/optimization){:.hackage} [](https://github.com/bgamari/optimization){:.github} : A number of optimization techniques from the modern optimization literature (quasi-Newton, stochastic gradient descent, mirror descent, projected subgradient etc.).
  - **sgd** [](https://hackage.haskell.org/package/sgd){:.hackage} [](https://github.com/kawu/sgd){:.github} : Stochastic gradient descent.




 

## Machine learning


### Bayesian inference
#### Nested sampling
  - **nested-sampling** [](http://hackage.haskell.org/package/NestedSampling){:.hackage} The code here is a fairly straightforward translation of the tutorial nested sampling code from Skilling and Sivia. The original code can be found at http://www.inference.phy.cam.ac.uk/bayesys/sivia/ along with documentation at http://www.inference.phy.cam.ac.uk/bayesys/. An example program called lighthouse.hs is included.
  - **NestedSampling-hs** [](https://github.com/eggplantbren/NestedSampling.hs){:.github} This is a Haskell implementation of the classic Nested Sampling algorithm introduced by John Skilling. You can use it for Bayesian inference, statistical mechanics, and optimisation applications, and it comes with a few example programs.
#### Frameworks
  - **probably-baysig** [](https://github.com/glutamate/probably-baysig){:.github} - This library contains definitions and functions for probabilistic and statistical inference.
    - Math.Probably.Sampler defines the sampling function monad
    - Math.Probably.PDF defines some common parametric log-probability density functions
    - Math.Probably.FoldingStats defines statistics as folds that can be composed and calculated independently of the container of the underlying data.
    - Strategy.\* implements various transition operators for Markov Chain Monte Carlo, including Metropolis-Hastings, Hamiltonian Monte Carlo, NUTS, and continuous/discrete slice samplers.
    - Math.Probably.MCMC implements functions and combinators for running Markov chains and interleaving transition operators.

#### Probabilistic programming languages
  - **monad-bayes** [](https://github.com/adscib/monad-bayes){:.github} - A library for probabilistic programming in Haskell using probability monads. The emphasis is on composition of inference algorithms implemented in terms of monad transformers. The code is still experimental, but will be released on Hackage as soon as it reaches relative stability. User's guide will appear soon. In the meantime see the models folder that contains several examples.
  - **hakaru** [](https://hackage.haskell.org/package/hakaru){:.hackage} [](https://github.com/hakaru-dev/hakaru){:.github} - Hakaru is a simply-typed probabilistic programming language, designed for easy specification of probabilistic models and inference algorithms. Hakaru enables the design of modular probabilistic inference programs by providing:
      - A language for representing probabilistic distributions, queries, and inferences
      - Methods for transforming probabilistic information, such as conditional probability and probabilistic inference, using computer algebra
  - **deanie** [](https://github.com/jtobin/deanie){:.github} - deanie is an embedded probabilistic programming language. It can be used to denote, sample from, and perform inference on probabilistic programs.



### Supervised learning

#### Time-series filtering
  - Kalman filtering
    - **estimator** [](https://hackage.haskell.org/package/estimator){:.hackage} The goal of this library is to simplify implementation and use of state-space estimation algorithms, such as Kalman Filters. The interface for constructing models is isolated as much as possible from the specifics of a given algorithm, so swapping out a Kalman Filter for a Bayesian Particle Filter should involve a minimum of effort.
This implementation is designed to support symbolic types, such as from sbv or ivory. As a result you can generate code in another language, such as C, from a model written using this package; or run static analyses on your model.
    - **kalman** [](https://hackage.haskell.org/package/kalman){:.hackage} Linear, extended and unscented Kalman filters are provided, along with their corresponding smoothers. Furthermore, a particle filter and smoother is provided.


#### Graphical models
  - Hidden Markov models
    - **HMM** [](http://hackage.haskell.org/package/HMM){:.hackage} [](https://github.com/mikeizbicki/hmm){:.github}
    - **hmm-hmatrix** [](http://hackage.haskell.org/package/hmm-hmatrix){:.hackage} [](http://hub.darcs.net/thielema/hmm-hmatrix){:.darcs} Hidden Markov Models implemented using HMatrix data types and operations. http://en.wikipedia.org/wiki/Hidden_Markov_Model 
It supports any kind of emission distribution, where discrete and multivariate Gaussian distributions are implemented as examples.
It currently implements:
      * generation of samples of emission sequences,
      * computation of the likelihood of an observed sequence of emissions,
      * construction of most likely state sequence that produces an observed sequence of emissions,
      * supervised and unsupervised training of the model by Baum-Welch algorithm.
    - **learning-hmm** [](http://hackage.haskell.org/package/learning-hmm){:.hackage} [](https://github.com/mnacamura/learning-hmm){:.github} This library provides functions for the maximum likelihood estimation of discrete hidden Markov models. At present, only Baum-Welch and Viterbi algorithms are implemented for the plain HMM and the input-output HMM.


#### Classification

  - Linear discriminant analysis
    - **linda** [](https://hackage.haskell.org/package/linda){:.hackage} LINDA implements linear discriminant analysis. It provides both data classification (according to Fisher) and data analysis (by discriminant criteria). Due to the `hmatrix` dependency, this package needs LAPACK installed, too. 
  - Support Vector Machines
    - **svm-simple** [](https://hackage.haskell.org/package/svm-simple){:.hackage} A set of simplified bindings to [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) suite of support vector machines. This package provides tools for classification, one-class classification and support vector regression.
      - Core Vector Machines
  - Decision trees
    - **hinduce-classifier-decisiontree** [](https://hackage.haskell.org/package/hinduce-classifier-decisiontree){:.hackage} [](https://github.com/roberth/hinduce-classifier-decisiontree){:.github}: A very simple decision tree construction algorithm; an implementation of `hinduce-classifier`'s Classifier class.
    - **HaskellGBM** [](https://github.com/dpkatz/HaskellGBM){:.github} : Haskell wrapper around LightGBM, the distributed library for gradient-boosted decision tree algorithms. The emphasis is on using Haskell types (in particular the `refined` library) to help ensure that the hyperparameter settings chosen by the user are coherent and in-bounds at all times.
  - Gaussian processes
    - **HasGP** [](https://hackage.haskell.org/package/HasGP){:.hackage} : Gaussian processes for regression and classification, based on the Laplace approximation and Expectation Propagation.

#### Neural Networks
  - **sibe** [](http://hackage.haskell.org/package/sibe){:.hackage} [](https://github.com/mdibaiee/sibe){:.github}
  - **neural** [](http://hackage.haskell.org/package/neural){:.hackage} [](https://github.com/brunjlar/neural){:.github}
  - **backprop-learn** [](https://github.com/mstksg/backprop-learn){:.github}
  - **grenade** [](http://hackage.haskell.org/package/grenade){:.hackage} [](https://github.com/HuwCampbell/grenade){:.github} Grenade is a composable, dependently typed, practical, and fast recurrent neural network library for precise specifications and complex deep neural networks in Haskell.
Grenade provides an API for composing layers of a neural network into a sequence parallel graph in a type safe manner; running networks with reverse automatic differentiation to calculate their gradients; and applying gradient decent for learning. Documentation and examples are available on github https://github.com/HuwCampbell/grenade.
  - **hasktorch** [](https://github.com/hasktorch/hasktorch){:.github} : Hasktorch is a library for tensors and neural networks in Haskell. It is an independent open source community project which leverages the core C libraries shared by Torch and PyTorch. This library leverages cabal new-build and backpack.
Note that this project is in early development and should only be used by contributing developers. Expect substantial changes to the library API as it evolves.
  - Recurrent Neural Networks
    - **grenade** [](http://hackage.haskell.org/package/grenade){:.hackage} [](https://github.com/HuwCampbell/grenade){:.github}
  - Convolutional Neural Networks
    - **grenade** [](http://hackage.haskell.org/package/grenade){:.hackage} [](https://github.com/HuwCampbell/grenade){:.github}
  - LSTM (Long Short-Term Memory)
    - **grenade** [](http://hackage.haskell.org/package/grenade){:.hackage} [](https://github.com/HuwCampbell/grenade){:.github}
    - **sibe** [](http://hackage.haskell.org/package/sibe){:.hackage} [](https://github.com/mdibaiee/sibe){:.github}
    - **neural** [](http://hackage.haskell.org/package/neural){:.hackage} [](https://github.com/brunjlar/neural){:.github}

  - Convolutional Neural Networks
    - **tensorflow** [](https://github.com/tensorflow/haskell){:.github} : Haskell bindings for TensorFlow
  
  - Generative Neural Networks
 
  - References : [Neural Networks, Types, and Functional Programming](https://colah.github.io/posts/2015-09-NN-Types-FP/)



#### Naive Bayes
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
    - **sibe** [](http://hackage.haskell.org/package/sibe){:.hackage} [](https://github.com/mdibaiee/sibe){:.github}
  - Bernoulli Naive Bayes



#### Boosting
  - XGBoost 
    - **xgboost-haskell** [](https://hackage.haskell.org/package/xgboost-haskell){:.hackage} XGBoost for Haskell, based on the foundation package. FFI binding of xgboost
    - **xgboost.hs** [](https://github.com/robertzk/xgboost.hs){:.github}
  - AdaBoost    

#### Regression

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
  

#### Reinforcement learning
  - **reinforce** [](https://github.com/sentenai-research/reinforce){:.github} `reinforce` exports an openai-gym-like typeclass, MonadEnv, with both an interface to [gym-http-api](https://github.com/openai/gym-http-api/), as well as haskell-native environments which provide a substantial speed-up to the http-server interface.
  - **gym-http-api** [](https://github.com/stites/gym-http-api){:github} [](https://hackage.haskell.org/package/gym-http-api){:hackage} This library provides a REST client to the gym open-source library. gym-http-api itself provides a python-based REST server to the gym open-source library, allowing development in languages other than python. Note that the openai/gym-http-api is a monorepo of all language-clients. This hackage library tracks stites/gym-http-api which is the actively-maintained haskell fork.
  - Policy gradient

  - Q-Learning
    - Neural Network Q-Learning

#### Clustering

  - K-Means
    - **kmeans** [](https://hackage.haskell.org/package/kmeans){:.hackage} [](http://hub.darcs.net/gershomb/kmeans){:.darcs} A simple implementation of the standard k-means clustering algorithm.
    - **clustering** [](https://hackage.haskell.org/package/clustering){:.hackage} Methods included in this library: Agglomerative hierarchical clustering: Complete linkage O(n^2), Single linkage O(n^2), Average linkage O(n^2), Weighted linkage O(n^2), Ward's linkage O(n^2). KMeans clustering.

  - Self-Organising Maps (SOM)
    - Hyperbolic-SOM
    - Hierarchical (H)SOMs

  - Mean-shift
  - Affinity propagation
  - Spectral Clustering
  - Hierarchical clustering
    - **clustering** [](https://hackage.haskell.org/package/clustering){:.hackage}
  - Birch

#### Dimensionality reduction

  - Principal Component Analysis (PCA)
    - **sibe** [](http://hackage.haskell.org/package/sibe){:.hackage} [](https://github.com/mdibaiee/sibe){:.github}
    - Kernel PCA
    - Incremental PCA
  - Truncated SVD*
  - Independent Component Analysis (ICA)
  - t-SNE (t-distributed stochastic neighbor embedding)
    - **tsne** [](https://hackage.haskell.org/package/tsne){:.hackage}
    
## Applications

  - Natural Language Processing (NLP)
    - **chatter** [](http://hackage.haskell.org/package/chatter){:.hackage} chatter is a collection of simple Natural Language Processing algorithms, which also comes with models for POS tagging and Phrasal Chunking that have been trained on the Brown corpus (POS only) and the Conll2000 corpus (POS and Chunking).
Chatter supports:
      - Part of speech tagging with Averaged Perceptrons. Based on the Python implementation by Matthew Honnibal: (http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/) See NLP.POS for the details of part-of-speech tagging with chatter.
      - Phrasal Chunking (also with an Averaged Perceptron) to identify arbitrary chunks based on training data.
      - Document similarity; A cosine-based similarity measure, and TF-IDF calculations, are available in the NLP.Similarity.VectorSim module.
      - Information Extraction patterns via (http://www.haskell.org/haskellwiki/Parsec/) Parsec
  - Bioinformatics
    - **NGLess** [](https://github.com/luispedro/ngless){:.github} Ngless is a domain-specific language for NGS (next-generation sequencing data) processing ("NGS with less work").


    

## Datasets

- **datasets** [](https://hackage.haskell.org/package/datasets){:.hackage} [](https://github.com/glutamate/datasets){:.github} - Classical machine learning and statistics datasets from the UCI Machine Learning Repository and other sources.
The datasets package defines two different kinds of datasets: 
  - Small data sets which are directly (or indirectly with file-embed) embedded in the package as pure values and do not require network or IO to download the data set. This includes **Iris**, **Anscombe** and **OldFaithful**
  - Other data sets which need to be fetched over the network and are cached in a local temporary directory.
- **mnist-idx** [](https://hackage.haskell.org/package/mnist-idx ){:.hackage} [](https://github.com/kryoxide/mnist-idx){:.github} - Read and write data in the IDX format used in e.g. the MINST database

## Language interop

### R

- HaskellR (https://tweag.github.io/HaskellR/)
  - **inline-r** [](https://hackage.haskell.org/package/inline-r){:.hackage} Seamlessly call R from Haskell and vice versa. No FFI required. Efficiently mix Haskell and R code in the same source file using quasiquotation. R code is designed to be evaluated using an instance of the R interpreter embedded in the binary, with no marshalling costs and hence little to no overhead when communicating values back to Haskell.
  - **H** [](https://hackage.haskell.org/package/H){:.hackage} An interactive prompt for exploring and graphing data sets. This is a thin wrapper around GHCi, with the full power of an R prompt, and the full power of Haskell prompt: you can enter expressions of either language, providing you with plotting and distributed computing facilities out-of-the-box.


# Machine learning misc.

  - **aima-haskell** [](https://github.com/chris-taylor/aima-haskell){:.github} Algorithms from Artificial Intelligence: A Modern Approach by Russell and Norvig. 



## Data science frameworks

### Apache Spark bindings

- **sparkle** [](https://hackage.haskell.org/package/sparkle){:.hackage} A library for writing resilient analytics applications in Haskell that scale to thousands of nodes, using Spark and the rest of the Apache ecosystem under the hood.
See the [blog post](https://www.tweag.io/posts/2016-02-25-hello-sparkle.html) for details: 
- **kraps-h** [](https://github.com/krapsh/kraps-haskell){:.github} : Haskell bindings to Apache Spark. The library consists of: 
  - A specification to describe data pipelines in a language-agnostic manner, and a communication protocol to submit these pipelines to Spark. 
  - A serving library, called krapsh-server, that implements this specification on top of Spark. It is written in Scala and is loaded as a standard Spark package.
  - A client written in Haskell that sends pipelines to Spark for execution. In addition, this client serves as an experimental platform for whole-program optimization and verification, as well as compiler-enforced type checking.


# Contribute

If you know a library that has to do with Data Science, please consider [adding](https://github.com/DataHaskell/docs/edit/gh-pages/_posts/2016-10-19-current-environment.md) it, if the category it belongs to doesn't exist, suggest a category for it.

Add sections related to data science and not only Machine Learning such as Data Mining, Distributed Processing, etc
