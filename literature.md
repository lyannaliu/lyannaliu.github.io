---
layout: page
permalink: /literature/index.html
title: Literature
pubs:
---
# Literature Review

## Contents

* [1. Predicting Song Popularity](#1)
* [2. The Million Song Dataset](#2)
* [3. Learning from Imbalanced Classes](#3)

This project reviews the literature relevant to its main purpose of identifying key characteristics associated with the likeability of songs. Articles considered for the review were mainly related to the use of the Million Song Dataset and the use of Python for the analysis, as well as all classification and prediction algorithms discussed during the CSCI 109 course, in Summer 2018 at Harvard University.

<h2 id="1">1. Predicting Song Popularity</h2>

Pham, J., Kyauk, E., and Park, E. (2015). Predicting Song Popularity. Department of Computer Science, Stanford University.

This analysis relates to predicting song popularity, motivated by its importance for the music industry competitiveness. The goal of this project was to determine what makes a song popular using the Million Song Dataset, which has a large number of audio features and metadata from one million songs. Different classification and regression algorithms predicted song popularity and determined the significant variables and their respective contributions.

The authors intended to determine if a song was popular or not based on song characteristics and singers. They used machine learning algorithms (SVMs, neural networks, logistic regression, Gaussian discriminant analysis, and linear regression) to output whether or not the song was popular. The predictors were artist familiarity, loudness, year, number of genre tags, and acoustic characteristics. 
 
The dataset for analysis was based on the Million Song Dataset, containing acoustic characteristics and several attributes. The authors extracted 10,000 tracks from the database, and after removing missing values and other irrelevant information, 2,717 tracks were used for analysis. 90% of the tracks were used for training and 10% for testing.
 
As discussed in the conclusion, the authors also created their own metric of popularity, defined as the number of downloads on iTunes or the number of plays on Spotify. The Lasso regression was their preferred method based on final findings.

<h2 id="2">2. The Million Song Dataset</h2>

Bertin-Mahieux, T., Daniel P. W., and Lamere, P. E. (2011). The Million Song Dataset. ISMIR. http://labrosa.ee.columbia.edu/millionsong/.
 
The Million Song Dataset is a collection of audio features and metadata for one million contemporary popular music tracks as presented by the authors. They reported on the dataset creation process, what was included in the dataset, and the dataset’s potential applications. The article contains resources and links to better understand the data and guide the user in analysis. The Million Song Dataset is the largest and most current research dataset in the music industry.

<h2 id="3">3. Learning from Imbalanced Classes</h2>
Fawcett, T. (2016) Learning from Imbalanced Classes. https://www.svds.com/learning-imbalanced-classes/

The article “Learning from Imbalanced Classes, August 25th, 2016”, discusses the goal of a classification algorithm to separate two categories and, particularly, the issues and approaches when dealing with imbalanced data; mathematical, statistical, or geometric assumptions may be considered when dealing with these classes.

Several datasets in machine learning are easy, mainly when the classes are balanced with similar number of observations in each class. Some datasets are just noisy or imbalanced.

In the article there is a picture with red points which are outnumbered by the blue colored observations. The imbalanced condition may be present on a diverse number of cases.  The article mentions as typical examples credit card fraud, medical conditions, disk failures and manufacturing defects. They may have distinct number of observations between classes.

Imbalanced data causes that conventional algorithms may be biased towards the majority class; their loss functions optimize quantities such as error rate. Minority examples could be treated as outliers of the majority class.

The article discusses situations of imbalanced classes in detail. In some datasets included by the Kaggle competition, the number of observations is fixed and no more data will be provided.

The blog shows code in Jupyter Notebook format with tutorials to solve these problems.

Imbalanced data has been studied during the last twenty years in machine learning. Efforts are reported in papers, workshops, special sessions, and dissertations. There are more than 220 references related to imbalanced data.

Among the alternatives about how to deal with imbalanced data are:

  - Do nothing. It may be enough to train the given data without need for modification.
  - Balance the training set with over-sampling the minority class, under-sampling the majority class or synthesizing a new minority class.
  - Throw away minority examples and switch to an anomaly detection framework.
  - Work with algorithms to adjust the class weight: adjust the decision threshold, modify an existing algorithm to be more sensitive to rare classes, or construct an entirely new algorithm to perform as desired on imbalanced data
  - Digression: evaluate what can be done and what cannot.

There are also some recommendations as follows:

  - Don’t use accuracy to evaluate a classifier.
  - Use probability estimates via proba or predict_proba and do not use hard labels.
  - In probability estimates, do not use 0.50 decision threshold to separate classes; a different value may be identified.
  - The article recommends the application of sklearn.cross_validation.StratifiedKFold to treat the natural given distributions and sklearn.calibration.CalibratedClassifierCV to avoid the use of probability estimates.
  - The article shows how to treat dimensionality of the data and the use of Cohen’s Kappa as an evaluation statistic on how much agreement would be expected just by chance.
  - Over-sampling and under-sampling techniques may be applied to force the data to be balanced. In this case, the variance in the classes must be understood. The machine learning community reports mixed results with over-sampling and under-sampling methods. The R package “unbalanced” implements methods to deal with imbalanced data.
  -  Use bagging to combine classifiers. This technique has not been implemented in Scikit-learn, with a file called blagging.py and BlaggingClassifier, which balances bootstrapped samples before aggregation.

Neighbor-based approaches are explained:

  - Over-sampling and under-sampling with randomly adjusted proportions. Other approaches examine the instance space carefully and decide what to do based on their neighborhoods.
  - Tomek’s algorithm looks for pairs and removes the majority instance of the pair.
  - The Chawla’s SMOTE (Synthetic Minority Oversampling TEchnique) system, to create more minority examples by interpolating between given observations.
  - Adjusting class weights with machine learning tools to implement importance of classes.

New algorithms efforts are included:

In 2014 Goh and Rudin wrote “Box Drawings for Learning with Imbalanced Data” showing two algorithms for learning from data with skewed examples. The method implements penalties with a form of regularization. They introduced two algorithms, one of which (Exact Boxes) uses mixed-integer programming and the other (Fast Boxes) with a clustering method. Experimental results show that both algorithms perform very well among a large set of test datasets.

Liu, Ting and Zhou developed a technique called Isolation Forests to identify anomalies in data using learning random forests with excellent results.

Next, Bandaragoda, Ting, Albrecht, Liu and Wells applied Nearest Neighbor Ensembles to improve Isolation Forests approach.

Finally, buy or create more data to construct examples of the rare class. This is an ongoing research in machine learning.
