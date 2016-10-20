---
layout: page
title: "[linear algebra] hmatrix"
category: library
date: 2016-10-20 03:22:38
---

hmatrix is a linear algebra library and matrix computations.

**Example code**:

```haskell

-- Creating matrices
r <- randn 2 2
r -- [  0.7668461757288327, 0.5573308002071669
  -- , -0.7412791132378888,  1.001032678483079 ]

-- SVD
(u, s, v) = svd r
(u, s, v) = thinSVD r

eigenvalues r

singularValues r

nullspace r
orthogonal r

determinant r
```

[**Benchmarks**](http://datahaskell.github.io/numeric-libs-benchmarks/benchmarks/hmatrix-linear-algebra.html)

**Notes**:

* Uses the [vector](/docs/library/vector) library under the hood (specifically, [`Data.Vector.Storable`](http://hackage.haskell.org/package/vector-0.11.0.0/docs/Data-Vector-Storable.html))

**Links**: [Hackage](http://hackage.haskell.org/package/hmatrix) . [GitHub](https://github.com/albertoruiz/hmatrix) . [Homepage](http://dis.um.es/~alberto/hmatrix/hmatrix.html)

