---
layout: page
---

# CS109A Data Science Final Project

<font color="#515151">Group #15: Haley Huang, Jose Garcia Crespo, Kezhou Miao, Liz Lawler</font><br /> 

## Motivation

Music Recommender Systems (MRS) have recently exploded in popularity thanks to music streaming services like Spotify, YouTube, and Last.fm. By some accounts, almost half of all current music consumption is through these services. MRS that can accurately interpret a user's song preferences and suggest suitable songs for a user can help keep users interested for longer periods of time and consequently bring in considerable profits. However, even large companies like Spotify, which has access to more than 35 million songs, 170 million monthly active users, and 75 million paying subscribers, use MRS that are far from perfect.

## Abstract

The goal of this study is to produce a model that, when given a song, predicts what songs a user will like by training and comparing its results against pre-existing playlists.

To achieve this goal, we built a model that is comprised of two sub-models and a metalearner. Both submodels use different datasets to produce a list of potential song recommendations and the probability each song is liked. The metalearner then uses the two predictions to produce a final list of song recommendations and an associated value of 1 if the song should be recommended or 0 if the song should not be recommended.

The first sub-model, named the Million Playlist model after its dataset, produced predictions that capture 8.6% of all songs on a pre-existing playlist when given one of the songs from the playlist. However, it also had a low precision rate of 11.9%, which means it recommended many songs that were not on the pre-existing playlist.

The second sub-model, named the Last.FM model after its dataset, produced predictions that capture .015% of the songs on a pre-existing playlist at a 8.5% precision rate. However, in contrast to the Million Playlist model, its local sensitivity was calculated at around 90%, which shows that it is better than the Million Playlist model at recognizing whether a song would be liked but oftentimes fails to consider a large enough set of potentially likable songs.

Although the expectation was for each model to cover for the weaknesses of the other when ensembled, this proved not to be the case. The final metalearner only successfully predicted .004% songs from pre-existing playlists given one of the songs at a precision of .64%. Given the relative strength of the sub-models, this was most likely due to poor tuning of the final metalearner, and we can expect stronger results with additional cross-validation of the metalearner parameters.
