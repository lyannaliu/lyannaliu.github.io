---
layout: page
permalink: /Appendix/index.html
title: Appendix
pubs:
---
# Appendix

This project reviews the literature relevant to its main purpose of identifying key characteristics associated with the likeability of songs. Articles considered for the review were mainly related to the use of the Million Song Dataset and the use of Python for the analysis, as well as all classification and prediction algorithms discussed during the CSCI 109 course, in Summer 2018 at Harvard University.

1. Pham, J., Kyauk, E., and Park, E. (2015). Predicting Song Popularity. Department of Computer Science, Stanford University.

This analysis relates to predicting song popularity, motivated by its importance for the music industry competitiveness. The goal of this project was to determine what makes a song popular using the Million Song Dataset, which has a large number of audio features and metadata from one million songs. Different classification and regression algorithms predicted song popularity and determined the significant variables and their respective contributions.

The authors intended to determine if a song was popular or not based on song characteristics and singers. They used machine learning algorithms (SVMs, neural networks, logistic regression, Gaussian discriminant analysis, and linear regression) to output whether or not the song was popular. The predictors were artist familiarity, loudness, year, number of genre tags, and acoustic characteristics. 
 
The dataset for analysis was based on the Million Song Dataset, containing acoustic characteristics and several attributes. The authors extracted 10,000 tracks from the database, and after removing missing values and other irrelevant information, 2,717 tracks were used for analysis. 90% of the tracks were used for training and 10% for testing.
 
As discussed in the conclusion, the authors also created their own metric of popularity, defined as the number of downloads on iTunes or the number of plays on Spotify. The Lasso regression was their preferred method based on final findings.

2.    Bertin-Mahieux, T., Daniel P. W., and Lamere, P. E. (2011). The Million Song Dataset. ISMIR. http://labrosa.ee.columbia.edu/millionsong/.
 
The Million Song Dataset is a collection of audio features and metadata for one million contemporary popular music tracks as presented by the authors. They reported on the dataset creation process, what was included in the dataset, and the dataset’s potential applications. The article contains resources and links to better understand the data and guide the user in analysis. The Million Song Dataset is the largest and most current research dataset in the music industry.
