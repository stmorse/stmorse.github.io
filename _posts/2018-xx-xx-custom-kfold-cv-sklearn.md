---
layout: post
title: "Custom upsampling for CV in scikit-learn"
categories: journal
date: 2018-07-15
tags: [projects, machine learning, python]
---

Imbalanced data can be a headache.  You have a beautiful labeled dataset, but the outcome you care about only constitutes <1% of the sample --- or less! Suddenly you have a host of new questions to answer: should you upsample/downsample? use weighting (and is this sometimes the same thing)? how should you measure prediction accuracy?

There is plenty elsewhere on this topic [ LINK LINK LINK ].  In this post I summarize some of these sources, and offer a quick hack for `sklearn` to help you incorporate some imbalanced data techniques into their ecosystem of methods and avoid hand-coding your pipeline.


# Imbalanced data strategies

The first thing to remember is: the baseline is already awesome.  You have 1,000 observations, with 10 positive.  Just predict "no" every time, and you are 99% accurate!  No machine learning required.  In fact, proceeding beyond this baseline implies that the minority class holds some extra importance: perhaps a rare medical condition, landmines, injuries, fraud ... something with an increased cost for misclassification, either false positives or false negatives (or both).  If not, then just predict "no"!  This realization drives the discussion.

**Sampling.**  One option is to up/down-sample the training data to create a 1:1 ratio --- this encourages the classifier to pay attention to the minority class.  Upsampling means generating more observations of the minority class, while downsampling means removing observations of the majority.  "Hybrid" methods do a little of both.  The most common way to upsample is sampling with replacement (or  *bootstrap*), although more sophisticated methods exist such as sampling from clusters, SMOTE, or [ other METHODS ].

Important caveat!  One should upsample *after* train/test split, not before; otherwise there will be bleed-over of test data in training.  See [ POST ]


**Weighting.** Another approach is to weight misclassification of the minority class more heavily, thus forcing the classifier to pay attention to it.

- Class weights in objective function.  When it is/is not the same as upsampling.  Effective learning rate issue with batch stochastic gradient.  #debate


**Boosting.**



Thresholding
- Confusion matrix, cost/loss function.  
- Note this is post-hoc, and doesn't affect training: your model is still just giving you the probability Class 1 occurs.  If it knew that a Type II error was more costly than a Type I, it wouldn't affect it's statement that a Class 1 is, say, 85% likely given this input.  [ CITE CITE ]



# Custom upsampling cross-validation

Standard k-fold cross-validation (CV) consists of partitioning the training set into $$k$$ equal parts, then: train on $$k-1$$ parts, test on the $$k$$th part (the "validation" set), then repeat on the next different subsetting.  Reason #1 we may do CV instead of just holding out a portion of the data as test data, is if we don't have enough data to afford this!  Reason #2 is we may use it to [ SELECT HYPERPARAMETERS ]  `sklearn` implements various flavors of this in the `model_selection` module, all essentially wrappers for generators returning indices to the splits.  For example their `KFold` class gives vanilla k-fold CV splits.

With imbalanced data this scheme creates a few wrinkles.  First, we probably want approximately equal representation of the minority class in each fold.  `sklearn` supports this with `StratifiedKFold`. Second, we may want to upsample/downsample 