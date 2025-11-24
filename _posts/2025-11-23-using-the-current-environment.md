---
layout: page
title: "Using the current environment"
category: getting_started
date: 2026-11-23 20:42:38
---

## Getting started

We recommend using **VS Code + Jupyter** as the default development stack for DataHaskell:
- VS Code as your editor
- Jupyter notebooks for literate, reproducible analysis
- A Haskell notebook kernel (currently IHaskell)
- The DataHaskell libraries (e.g. `dataframe`, `hasktorch`, plotting, etc.)

This page walks you through:

1. Installing the basic tools  
2. Choosing an environment (Dev Container vs local install)  
3. Verifying everything with a “hello DataHaskell” notebook  

---

## 1. Install the basics

You only need to do this once per machine.

### 1.1. VS Code

1. Install **Visual Studio Code** from the official website.
2. Open VS Code and install these extensions:
   - **Jupyter**
   - **Python** (used by the Jupyter extension, even if you write Haskell)
   - **Dev Containers** (if you plan to use the container-based environment)
   - **Haskell** (for syntax highlighting, type info, etc.)

### 1.2. Git

Install Git so you can clone repositories:

- macOS: via Homebrew (`brew install git`) or Xcode command line tools  
- Linux: via your package manager (e.g. `sudo apt install git`)  
- Windows: [Git for Windows] or via WSL (Ubuntu on Windows)

### 1.3. (Optional but recommended) Docker

If you want the easiest, most reproducible setup, install Docker:

- Docker Desktop (macOS/Windows) or  
- `docker` + `docker-compose` from your Linux distro

The Dev Container–based environment assumes Docker is available.

---

## 2. Choose an environment

You have **two main options**:

1. **Option A (recommended): VS Code Dev Container**  
   Everything is pre-installed in a Docker image (GHC, Cabal/Stack, IHaskell, DataFrame, etc).

2. **Option B: Local installation**  
   Install GHC, Cabal, Jupyter, IHaskell, and DataHaskell libraries directly on your machine.

If you’re not sure which to choose, pick **Option A**.

---

## 3. Option A – Dev Container (recommended)

This is the “batteries included” path. You get a pinned environment without polluting your global system.

### 3.1. Clone the starter repository

We provide a starter repository with a ready-made environment and example notebooks:

```bash
git clone https://github.com/DataHaskell/datahaskell-starter
cd datahaskell-starter
```

### 3.2. Open the project in VS Code

```bash
code .
```

You'll get a popup asking if you want to re-ooen the project in a container.
Select this option and VS Code will load the DataHaskell docker file.

### 3.3. Running the example notebook

Open the `getting-started` notebook. You'll see a section that says `Select Kernel` at the top right.

Upon clicking it you'll be asked to select a kernel. Go to `Jupyter Environment` and use the Haskell kernel installed there.

## 3. Option B – Installing everything locally

We recommend you use cabal for this section.

```bash
cabal update
cabal install --lib dataframe ihaskell-dataframe hasktorch \
    ihaskell dataframe-hasktorch ihaskell-dataframe time ihaskell template-haskell \
    vector text containers array random unix directory regex-tdfa containers \
    cassava statistics monad-bayes aeson \
    --force-reinstalls
cabal install ihaskell --install-method=copy --installdir=/opt/bin
ihaskell install --ghclib=$(ghc --print-libdir) --prefix=$HOME/.local/
jupyter kernelspec install $HOME/.local/share/jupyter/kernels/haskell/
jupyter notebook
```

Check if this setup is working by trying out the linear regression tutorial from the DataHaskell website.

> Note this way of globally installing packages might break some of your existing projects.

