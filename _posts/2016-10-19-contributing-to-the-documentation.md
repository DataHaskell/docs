---
layout: page
title: "Contributing to the documentation"
category: community
date: 2016-10-19 15:43:05
---
**dataHaskell**'s documentation is served as a Jekyll site, using GitHub pages.  

## Steps for contribution

1. **Fork** the [dataHaskell/docs](https://github.com/DataHaskell/docs) repository.
2. If you want to add a page, in the root of the repository **execute** `ruby bin/jekyll-page "TITLE" CATEGORY` where `TITLE` is the title of the page that you want to add, and `CATEGORY` is one of these:
  - `community` - Documentation related to the community. Contribution guidelines, codes of conduct, information of interest, events...
  - `tutorial` - Tutorial on how to achieve different data science or Haskell goals. From doing a regression, to understanding how to SubHask works.
  - `library` - Library overviews. Benchmarks, documentation for them (if no good official documentation is provided), advantages and disadvantages.
  - `other` - Before submitting any page to this category, discuss with the community if a new category should be created.
3. Optionally, but optimally, do a `jekyll serve` in the root of the repository to be sure that all the contributions you've made are displayed correctly.

## Things to have in mind

- Where possible, adhere to the existing formats. For example, if adding a library overview try to mimick as much as possible the format of other pages in the same category. If you are submitting something new, ask in the community.
- When writing informative material, which will be most of the time, try to explain as for a person that does not know anything but has the ability of learning everything in a split second, avoiding acronyms. For example, instead of saying: *"Here we use a DRNN for..."* try to write *"Here we use a Deep Recursive Neural Network (which is a kind of machine learning algorithm) for..."*. Even if this is not 100% true, it might save the reader a couple of minutes googling and figuring out how/why it works. The reader might deepen into the subject if it is their desire.
- **Use a lot of examples**, even if they are really simple. Most of the people learn better from examples than from theorems or just words. If you are able to, include graphics, plots, code snippets or whatever comes to your mind.
