---
layout: page
title: "Getting started with HaskellDO"
category: tutorial
date: 2016-10-20 21:35:06
---

*This page is based on the original `README.md` file for [Theam's HaskellDO repo](https://github.com/theam/haskell-do) *

HaskellDO is a Haskell code editor, centered around interactive development.

## Usage

The only *3rd-party* requirement to run HaskellDO is [Stack](http://haskellstack.org/) and [NodeJS](https://nodejs.org/).

Before proceeding, run a `npm install -g purescript pulp bower` to install the required NodeJS binaries.

**Clone** this repository, and from the root of the project run:

`make deps` for installing the required dependencies, and

`make build-all-<platform>`

Where `<platform>` is one of:

- `windows`
- `linux`
- `osx`

Choose accordingly to your platform.

### Initializing a project
Begin by creating a **new** Stack project.

`stack new your_project_name`

Fire up HaskellDO by running `make run` from the root of the project,
it will ask you to open a Stack project.
Navigate to the root of the project you just created and open that
folder.

### Main interface
Let's begin by adding a text cell for documenting our analysis:

![Imgur](http://i.imgur.com/QAVI2WC.gif)

HaskellDO's text editor is based on [SimpleMDE](https://simplemde.com/) for
handling the editing and rendering of the "documentation" part of our code.

It supports all the features that you would expect from a normal markdown
editor, including image embedding.

- Do a single click out of the editor to render the current text
- Do a double click inside of the editor to come back to the text editing
  view.

![Imgur](http://i.imgur.com/ElGTVLK.gif)

Now, **it's time to work with some code, let's insert a code cell**.
In it, we write regular Haskell code, like we would do in a normal Haskell
module.

In fact, our whole page is just a Haskell file that can be used in any
Haskell, project. No need to export/import!

![Imgur](http://i.imgur.com/8jVxh6A.gif)

Now we have to try our new algorithm we spent hours researching on.
No, we do not have to spawn a `stack ghci` or a `stack repl`. HaskellDO
manages that for us and reloads the code each time we ask it to evaluate
some expression.

Just click on the **toggle console** button and press return on it to
enable it.

After writing your expression, press return [twice](https://github.com/theam/haskell-do/issues/1)
to get the result written on screen.

![Imgur](http://i.imgur.com/jgZQAvu.gif)

But, what does our *real* module file look like? Let's see the contents
of our `Main.hs` file:

```haskell
-- # Analyzing dog cuteness with genetic algorithms
--
-- After going **through thorough and tough thoughts**, we decided to use a simple example.

a = [1..20]
b = f <$> a

f :: Int -> Int
f x = x * 3 + 4
```

Just regular Haskell! Ready for deployment!

