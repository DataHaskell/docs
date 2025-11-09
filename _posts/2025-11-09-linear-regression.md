---
layout: page
title: "Linear Regression: California House Price Prediction"
category: tutorial
date: 2025-11-09 14:08:18
---

In this tutorial, we'll predict California housing prices using two awesome libraries: **DataFrame** (for data wrangling) and **Hasktorch** (for machine learning).

## What Are We Building?

We're going to:
1. 📊 Load and clean real housing data
2. 🔧 Engineer some clever features
3. 🤖 Train a linear regression model
4. 🎯 Predict house prices!

Think of it as teaching a computer to estimate home values based on things like location, number of rooms, and how close the house is to the ocean.

## Our libraries

### DataFrame
DataFrame is Swiss Army knife of data manipulation. It lets you work with tabular data (like CSV files) in a mostly type-safe, functional way.

### Hasktorch
Hasktorch brings the power of Torch to Haskell. It lets us do numerical computing and machine learning. It has tensors (multi-dimensional arrays) which are the building blocks of neural networks.

## Let's Dive Into The Code!

### Setting Up Our Imports

```haskell
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified DataFrame as D
import qualified DataFrame.Functions as F
import DataFrame.Hasktorch (toTensor)
import Torch
import DataFrame ((|>))
```

**What's happening here?** We're enabling some handy language extensions and importing our tools. The `|>` operator is particularly cool. The operator is like the Unix pipe, letting us chain operations left-to-right!

### Step 1: Loading the Data 📂

```haskell
df <- D.readCsv "../data/housing.csv"
```

**Simple, right?** We're loading California housing data from a CSV file. This dataset contains information about different neighborhoods—things like population, median income, and (importantly) median house values.

### Step 2: Handling Missing Data 🔍

Real-world data is messy. Sometimes values are missing, and we need to deal with that:

```haskell
let meanTotalBedrooms = df |> D.filterJust "total_bedrooms" |> D.mean
```

**Translation:** "Hey DataFrame, take our data, filter out the rows where `total_bedrooms` is missing, then calculate the mean of what's left."

We'll use this mean to fill in the blanks later. This is called **imputation**—fancy word for "educated guess filling."

### Step 3: Feature Engineering

Arguably the most important part of the learning process is making sure your data is meaningful. This is called **"feature engineering."** We want to combine our features in interesting ways so that patterns become easier for the model to spot.

Machine learning models are powerful, but they're not magic. They can only learn from what we give them. If we just hand over raw numbers, we're making the model work way harder than it needs to. But if we do some creative thinking and craft features that highlight the relationships we care about, we can make even a simple model perform amazingly well.

In our housing example, we're going to:
- Convert text categories (like "NEAR OCEAN") into numbers the model can use (with 0 being the closest to the ocean and 5 being the furthest).
- Create a brand new feature: `rooms_per_household` (because maybe spacious homes are worth more?)
- Normalize everything so no single feature dominates

```haskell
let cleaned =
        df
            |> D.impute (F.col @(Maybe Double "total_bedrooms")) meanTotalBedrooms
            |> D.exclude ["median_house_value"]
            |> D.derive "ocean_proximity" (F.lift oceanProximity (F.col "ocean_proximity"))
            |> D.derive
                "rooms_per_household"
                (F.col @Double "total_rooms" / F.col "households")
            |> normalizeFeatures

oceanProximity :: T.Text -> Double
oceanProximity op = case op of
    "ISLAND" -> 0
    "NEAR OCEAN" -> 1
    "NEAR BAY" -> 2
    "<1H OCEAN" -> 3
    "INLAND" -> 4
    _ -> error ("Unknown ocean proximity value: " ++ T.unpack op)
```

**Let's break this pipeline down:**

1. **Impute**: Fill in those missing bedroom values with the mean we calculated
2. **Exclude**: Remove the house value column (we'll use it as labels, not features)
3. **Derive ocean_proximity**: Convert text like "NEAR OCEAN" into numbers (0-4) that our model can understand
4. **Derive rooms_per_household**: Create a new feature! Maybe houses with more rooms per household are worth more?
5. **Normalize**: Scale all features to a 0-1 range so no single feature dominates

### Feature Normalization

```haskell
normalizeFeatures :: D.DataFrame -> D.DataFrame
normalizeFeatures df =
    df
        |> D.fold
            ( \name d ->
                let col = F.col @Double name
                 in D.derive name ((col - F.minimum col) / (F.maximum col - F.minimum col)) d
            )
            (D.columnNames (df |> D.selectBy [D.byProperty (D.hasElemType @Double)]))
```

Neural networks do better when all the data is scaled the same. We applying **min-max normalization** to every numeric column:

```
normalized_value = (value - min) / (max - min)
```

This squishes every feature to the 0-1 range. Why? Imagine if house prices ranged from 0-500,000 but number of bedrooms ranged from 0-5. The huge price numbers would dominate the small bedroom numbers during training. Normalization levels the playing field.

### Step 4: From DataFrame to Tensors 🔢

```haskell
features = toTensor cleaned
labels = toTensor (D.select ["median_house_value"] df)
```

**Bridge time!** We're converting our nice, clean DataFrame into Hasktorch tensors. Think of tensors as supercharged matrices that GPUs love to work with. Our `features` are what the model learns from, and `labels` are what it's trying to predict.

### Step 5: Building Our Model

```haskell
init <- sample $ LinearSpec{in_features = snd (D.dimensions cleaned), out_features = 1}
```

**What's a linear model?** Imagine drawing the best-fit line through a scatter plot—except we're doing it in many dimensions. The model learns:

```
house_price = w₁×feature₁ + w₂×feature₂ + ... + wₙ×featureₙ + bias
```

We're creating a linear layer with as many inputs as we have features (after cleaning) and 1 output (the predicted price).

```haskell
model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input
```

This is our prediction function—feed in features, get out a price estimate.

### Step 6: Training Loop

```haskell
trained <- foldLoop init 100_000 $ \state i -> do
    let labels' = model state features
        loss = mseLoss labels labels'
    when (i `mod` 10_000 == 0) $ do
        putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    (state', _) <- runStep state GD loss 0.1
    pure state'
```

**This is where learning happens!** Let's break it down:

1. **100,000 iterations**: The model gets 100,000 chances to improve
2. **labels'**: Make predictions with current model weights
3. **loss**: How wrong are we? MSE (Mean Squared Error) measures the average squared difference between predictions and real prices
4. **Print every 10,000 steps**: Show us how we're doing!
5. **runStep with GD**: Update the model using **Gradient Descent** with a learning rate of 0.1
   - Think of gradient descent as rolling a ball down a hill to find the lowest point (best model)
   - Learning rate controls how big our steps are

**What you'll see:**
```
Training linear regression model...
Iteration: 10000 | Loss: Tensor Float []  5.0225e9   
Iteration: 20000 | Loss: Tensor Float []  4.9093e9   
Iteration: 30000 | Loss: Tensor Float []  4.8576e9   
Iteration: 40000 | Loss: Tensor Float []  4.8333e9   
Iteration: 50000 | Loss: Tensor Float []  4.8217e9   
Iteration: 60000 | Loss: Tensor Float []  4.8160e9   
Iteration: 70000 | Loss: Tensor Float []  4.8130e9   
Iteration: 80000 | Loss: Tensor Float []  4.8114e9   
Iteration: 90000 | Loss: Tensor Float []  4.8105e9   
Iteration: 100000 | Loss: Tensor Float []  4.8099e9 
```

### Step 7: Making Predictions

```haskell
let predictions =
        D.insertUnboxedVector
            "predicted_house_value"
            (asValue @(VU.Vector Float) (model trained features))
            df
print $ D.select ["median_house_value", "predicted_house_value"] predictions
```

**The grand finale!** We're:
1. Using our trained model to predict all the house values
2. Converting the tensor back to a vector
3. Adding it as a new column in our original DataFrame
4. Printing a comparison of real vs. predicted values

You'll see something like:
```
-------------------------------------------
 median_house_value | predicted_house_value
--------------------|----------------------
       Double       |         Float        
--------------------|----------------------
 452600.0           | 414079.94            
 358500.0           | 423011.94            
 352100.0           | 383239.06            
 341300.0           | 324928.94            
 342200.0           | 256934.23            
 269700.0           | 264944.84            
 299200.0           | 259094.13            
 241400.0           | 257224.55            
 226700.0           | 201753.69            
 261100.0           | 268698.7
...
```

## Key Concepts We Learned

**DataFrame Operations:**
- `|>` - Pipeline operator (read left to right!)
- `readCsv` - Load data from CSV files
- `impute` - Fill in missing values
- `derive` - Create new columns from existing ones
- `filterJust` - Remove rows with missing values
- `select` / `exclude` - Choose which columns to keep

**Hasktorch:**
- `toTensor` - Convert DataFrames to tensors
- `Linear` - Linear regression layer
- `mseLoss` - Mean Squared Error loss function
- `runStep` with `GD` - Gradient descent optimization
- `sample` - Initialize model parameters

**Machine Learning Flow:**
1. **Load Data** → Get it into your program
2. **Clean & Transform** → Handle missing values, normalize
3. **Feature Engineering** → Create useful new features
4. **Train** → Iteratively improve the model
5. **Predict** → Use the trained model on data

## Try It Yourself!

**Experiment ideas:**
- Change the learning rate (0.1) to see how it affects training
- Add more derived features (like income per person)
- Try different numbers of iterations
- Use different normalization strategies

## Why This Approach Rocks 🌟

1. **Type Safety**: DataFrame's type system catches most errors at compile time
2. **Functional Style**: Pure functions and pipelines make data transformations clear
3. **Performance**: Hasktorch uses PyTorch's battle-tested backend
4. **Readability**: The `|>` operator makes data pipelines read like stories

## Next Steps

Now that you've mastered the basics:
- Try different models (polynomial regression, neural networks)
- Experiment with more complex feature engineering
- Learn about train/test splits and model validation
- Explore Hasktorch's neural network modules
