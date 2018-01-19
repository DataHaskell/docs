---
layout: page
title: "basic setup and beginner tutorials"
category: tutorial
date: 2018-01-18 13:14:00
---

## Dependencies

If you don't already have a set workflow for Haskell development, may I suggest

- [stack](https://haskell-lang.org/get-started)

    User-friendly tool to set up projects and run builds.

- [emacs](https://www.gnu.org/software/emacs/)

    A battle-hardened and featureful editor/IDE. Haskell tooling plays very nicely with it.

- [intero](https://haskell-lang.org/intero)

    Haskell interactive development extension for IDEs

## Basic workflow

- [idiomatic haskell](https://github.com/Gabriel439/slides/blob/master/lambdaconf/data/data.md)


    Gabriel Gonzalez guides us through a simple data science workflow using a selection of basic libraries. In his own words, the state of the ecosystem is that

    > All the tools are there, but not yet organized into a cohesive whole
    
    *If you are not a [nix](https://nixos.org/) user, I suggest you create your own project and add his code.*

- [reading a csv](http://howistart.org/posts/haskell/1/) 

    Chris Allen has written a very approachable tutorial describing all the basic workflow from creating a Haskell project to reading a CSV file to summing how many *at bats* present in the file. It also showcases something that Haskell lets you more conveniently than other languages: streaming the data in constant memory.

- [writing to a csv](https://www.stackbuilders.com/tutorials/haskell/csv-encoding-decoding/) 

    Juan Pedro Villa Icaza wrote another tutorial that includes *writing* to a CSV.

