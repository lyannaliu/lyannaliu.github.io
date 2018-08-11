---
layout: page
permalink: /modelsandmethods/index.html
title: Models and Methods
---

# Models and Methods

## Contents

* [1. Overview of the Models](#1)
* [2. Million Playlist Model](#2)
    * [2.1 Data Preparation](#2.1)
    * [2.2 Logistic Regression](#2.2)
    * [2.3 Decision Tree](#2.3)
    * [2.4 Neural Network](#2.4)
    * [2.5 Ensemble Method](#2.5)
* [3. Last.FM Model](#3)
    * [3.1 Data Preparation](#3.1)
    * [3.2 Logistic Regression](#3.2)
    * [3.3 Bagging](#3.3)
    * [3.4 Boosting](#3.4)
    * [3.5 Neural Network](#3.5)
    * [3.6 Ensemble Model](#3.6)
* [4. Metalearner](#4)

<h2 id="1">1. Overview of the Models</h2>
The recommendation model is a metalearner that ensembles two submodels (Million Playlist Model and Last.FM Model), which are the ensemble of three to four sub-submodels which are trained directly on the two submodels' respective processed datasets. 

![Fig](/images/fig3_diagram.png)

We chose this structure for several reasons:

1. The Last.FM and Million Playlist datasets are very different. The Last.FM dataset is missing newer songs, whereas the Million Playlist dataset includes very recent songs.

2. We wanted to observe how various types of models performed against the two datasets and compare the submodels' performances to the performance of ensembled models.

3. Given the unique nature of the data, we were not sure which type of models would perform the best, so instead we chose to select several and use metalearners to determine the strengths and weakenesses of each model. 

<h2 id="2">2. Million Playlist Model</h2>
The Million Playlist Model will be an ensembled model of three sub-models and a metalearner. The metalearner is AdaBoost and  the three sub-models are as follows:
- Logistic Regression
- Decision Tree
- Neural Network

All three sub-models predict the probability of a song being a 'hit' (a song that appears on the target test playlist) and are then fed into an Adaboost metalearner model that combines the three predictions into a final prediction. The result of the AdaBoost metalearner will then feed into the final ensembler model - along with the output from the Last.FM model - to produce a final list of recommended songs.
<h3 id="2.1">2.1 Data Preparation</h3>
The input into the Million Playlist Model should be a randomly selected song (called the `target track`) from a randomly selected playlist (called the `target playlist`). Given the `target track`, a dataset is dynamically generated from the 900,000 training playlists by building a set of tracks that are `related` to the `target track`. `Related track`s are either directly related to the `target track`, have a `related artist`, or is in a `related album`.

- A `related track` is defined as a track that is not the `target track` but appears in the same playlist as the `target track`.
- A `related artist` is defined as an artist that is not the `target track`'s artist but appears in the same playlist as the `target track`.
- A `related album` is defined as an album that is not the `target track`'s album but appears in the same playlist as the `target track`.

For each `related track`, the following attributes are calculated:

- Related track frequency: the number of times the `related track` appears in the same playlist as the `target track`.
- Related artist frequency: the number of times the `related artist` appears in the same playlist as the `target track`.
- Related album frequency: the number of times the `related album` appears in the same playlist as the `target track`.
- Total song frequency: the number of times the `related track` appears across all playlists.
- Total artist frequency: the number of times the `related track`’s artist appears across all playlists.
- Total album frequency: the number of times the `related track`’s album appears across all playlists.

For example, for the `target track` "Piano Man", the following dataset is dynamically generated:

![Piano Man DF](/images/piano_man_df.png)

In this example, the `related track` "Ghosts 'n' Stuff" by deadmau5 from the album "For Lack of a Better Name" was found 4 times in the same playlist as "Piano Man", the artist deadmau5 showed up 24 times in the same playlist as "Piano Man", and the album "For Lack of a Better Name" showed up 6 times in the same playlist as "Piano Man". Overall, the song "Ghosts 'n' Stuff" showed up in 332 in playlists, the artist deadmau5 showed up 3259 times in playlists, and the album "For Lack of a Better Name" showed up 592 times in playlists.

Several different functions come together to generate these datasets.

First, dictionaries that relate artists and their songs and albums and their songs are created. This is a one-time run.

```python
def create_artist_album_lists(allDetails):
    artistList = {}
    albumList = {}
    for track in allDetails:
        if track.artist in artistList:
            artistList[track.artist].append(track)
        else:
            artistList[track.artist] = [track]

        if track.album in albumList:
            albumList[track.album].append(track)
        else:
            albumList[track.album] = [track]

    return (artistList, albumList)

artist_song_list, album_song_list = create_artist_album_lists(songDetails)
```

`artist_song_list` is a dictionary where the key is an artist and the value is a list of all of the artist's tracks.

```python
{'Alexandre Desplat': [Track(song='Dandelions', artist='Alexandre Desplat', album='Afterwards (Original Motion Picture Soundtrack)'),
  Track(song='The Imitation Game', artist='Alexandre Desplat', album='The Imitation Game (Original Motion Picture Soundtrack)'),
  Track(song='Circles', artist='Alexandre Desplat', album='The Tree of Life (Original Motion Picture Soundtrack)'),
  Track(song="Alan Turing's Legacy", artist='Alexandre Desplat', album='The Imitation Game (Original Motion Picture Soundtrack)'),
  ...],
  'Billy Joel': [Track(song='And So It Goes', artist='Billy Joel', album='Storm Front'),
  Track(song='Piano Man', artist='Billy Joel', album='Piano Man'),
  Track(song="It's Still Rock and Roll to Me", artist='Billy Joel', album='Glass Houses'),
  Track(song='Uptown Girl', artist='Billy Joel', album='An Innocent Man'),
  Track(song="We Didn't Start the Fire", artist='Billy Joel', album='Storm Front'),
  Track(song='Only the Good Die Young', artist='Billy Joel', album='The Stranger (30th Anniversary Legacy Edition)'),
  Track(song='My Life', artist='Billy Joel', album='52nd Street'),
  ...],
  ...
}
```

`album_song_list` is a dictionary where the key is an album and the value is a list of all the tracks in the album.

```python
{'Piano Man': [Track(song='Piano Man', artist='Billy Joel', album='Piano Man'),
 Track(song='Piano Man', artist='Mamamoo', album='Piano Man'),
 Track(song='Captain Jack', artist='Billy Joel', album='Piano Man'),
 Track(song="You're My Home", artist='Billy Joel', album='Piano Man'),
 Track(song='The Ballad of Billy the Kid', artist='Billy Joel', album='Piano Man'),
 Track(song='If I Only Had the Words (To Tell You)', artist='Billy Joel', album='Piano Man'),
 ...],
 'The Imitation Game (Original Motion Picture Soundtrack)': [Track(song='The Imitation Game', artist='Alexandre Desplat', album='The Imitation Game (Original Motion Picture Soundtrack)'),
 Track(song="Alan Turing's Legacy", artist='Alexandre Desplat', album='The Imitation Game (Original Motion Picture Soundtrack)'),
 Track(song='Alan', artist='Alexandre Desplat', album='The Imitation Game (Original Motion Picture Soundtrack)'),
 Track(song='Farewell to Christopher', artist='Alexandre Desplat', album='The Imitation Game (Original Motion Picture Soundtrack)'),
 Track(song='End of War', artist='Alexandre Desplat', album='The Imitation Game (Original Motion Picture Soundtrack)'),
 ...],
 ...
 }
```

Any time a dataset needs to be created for a song, the following function is run to generate the Related Track Frequency, Related Artist Frequency, and Related Album Frequency attributes.

 ```python
# Get frequency of other songs, artists, and albums in other playlists compared to a base track
# To get additional options, for each artist, add all the artist's songs (that aren't already added)
# and for each album, add all the album's songs (that aren't already added) and give them counts of 0
# (since they must not have shown up as related in any of the playlists)
def get_related_tracks(base_track, playlist_list):
    countSongs = Counter()
    countArtists = Counter()
    countAlbums = Counter()

    # for each playlist in our training set
    for playlist in playlist_list:
        # check if the user provided track is any of the training playlists
        # if it is, update the total number of times we see all the other songs in the playlist
        if base_track in playlist:
            # then for each song related to the base_track, count the frequency
            # of the artists and albums that also appear in the playlist
            for track in playlist:
                if track != base_track:
                    countSongs[track] += 1
                    countArtists[track.artist] += 1
                    countAlbums[track.album] += 1
    
    # also add every related artists' and albums' songs that haven't already been added
    for artist in countArtists:
        for track in artist_song_list[artist]:
            if track not in countSongs:
                countSongs[track] = 0
    for album in countAlbums:
        for track in album_song_list[album]:
            if track not in countSongs:
                countSongs[track] = 0
                
    return (countSongs, countArtists, countAlbums)
```

Running this function for "Piano Man" returns three counters:

```python
piano_man_song_count, piano_man_artist_count, piano_man_album_count = \
    get_related_tracks(piano_man, detailed_train_playlists)

# piano_man_song_count
Counter({Track(song='Little Lion Man', artist='Mumford & Sons', album='Sigh No More'): 35,
         Track(song="Ghosts 'n' Stuff", artist='deadmau5', album='For Lack of a Better Name'): 4,
         Track(song='i', artist='Kendrick Lamar', album='i'): 5,
         Track(song='No Light, No Light', artist='Florence + The Machine', album='Ceremonials'): 4,
         ...

# piano_man_artist_count
Counter({'Mumford & Sons': 299,
         'deadmau5': 24,
         'Kendrick Lamar': 190,
         'Florence + The Machine': 119,
         ...

# piano_man_album_count
Counter({'Sigh No More': 117,
         'For Lack of a Better Name': 6,
         'i': 5,
         'Ceremonials': 37,
         ...
```

The full dataframe is then built using the next function, `create_a_test_df`. All of the tracks from the `target track`'s `target playlist` are assigned a value of 1 in the "hit" attribute. All other tracks are assigned a value of 0. In most situations, this results in dataframes that are almost entirely comprised of "misses" since there can be thousands of `related track`s but only a hundred or songs in the `target playlist`. This is accounted for later by assigning class weights to the models.

```python
# Checks if a track is in the target playlist
def checkForHit(song, target_playlist):
    if song in target_playlist:
        return 1
    else:
        return 0

# Create a dataframe that contains *all* related songs (instead of a subset with a trimmed number of 'misses')
def create_a_test_df(target_playlist, target_song=None):
    # select a random song from the playlist if none is provided
    if target_song == None:
        target_song = random.choice(target_playlist)
    # create all the frequency counts for the target song
    songList, artistList, albumList = get_related_tracks(target_song, detailed_train_playlists)
    # column names for the dataframe
    column_names = ['related_song_frequency', 'related_artist_frequency', 'related_album_frequency', \
                'total_song_frequency', 'total_artist_frequency', 'total_album_frequency', 'hit']
    # create list of song data
    list_for_df = {song : [songList[song],
                  artistList[song.artist],
                  albumList[song.album],
                  songDetails[song], 
                  totalArtistCount[song.artist], 
                  totalAlbumCount[song.album],
                  checkForHit(song, target_playlist)] for song in songList}
    
    # We need to check for a case where there are only 'hits' in the dataframe
    # If the length of the target playlist is shorter than the total songList, create the dataframe normally
    if len(target_playlist) < len(songList):
        df = pd.DataFrame.from_dict(list_for_df, orient='index', columns=column_names)
    else: # If there are no related 'misses' (class 0 values) then just pull random songs 
        print('length of target playlist = length of related song list')
        random_list_for_df = get_random_song_list(len(target_playlist), target_playlist, artistList, albumList)
        random_list_for_df.update(list_for_df)
        df = pd.DataFrame.from_dict(random_list_for_df, orient='index', columns=column_names)
        
    return df, target_song
```

Note that in the case where a `target track` has no `related track`s, tracks are randomly selected from random playlists in the detailed_train_playlists list and added to the dataframe. This is to avoid a scenario where a dataset only contains "hits".

```python
# If there are NO related songs, artists, or albums, pick a random set of songs size equal to the number of 'hits'
# from the overall playlist to use as 'misses'
def get_random_song_list(length, target_playlist, artistList, albumList):
    column_names = ['related_song_frequency', 'related_artist_frequency', 'related_album_frequency', \
            'total_song_frequency', 'total_artist_frequency', 'total_album_frequency', 'hit']
    random_song_list = random.sample(list(songDetails), length)
    # We can assign relationships as 0 because otherwise they would have been picked up already
    # Still check for a hit just in case one of the target songs were selected
    # TODO: 
    random_list_for_df = {song : [0,
                                  artistList[song.artist],
                                  albumList[song.album],
                                  songDetails[song], 
                                  totalArtistCount[song.artist], 
                                  totalAlbumCount[song.album],
                                  checkForHit(song, target_playlist)] for song in random_song_list}

    return random_list_for_df
```

The final result is a dataframe containing the dynamically calculated frequency values for the `target song`. As shown earlier, for Piano Man, this dataframe would look something like the dataframe below.


![Piano Man DF](/images/piano_man_df.png)

<h3 id="2.2">2.2 Logistic Regression</h3>
The logistic regression submodel is an ensembled set of eight logistic models. The eight models were trained separately on eight randomly selected `target tracks` and `target playlists` from the detailed_train_playlists set. 0's were given a class weight of .11 versus the 1's which were given a class weight of .89 to adjust for the large number of misses in the dataset versus the small number of hits. This number seemed to be most successful in our limited number of tries. The final prediction is either the majority if producing a binary result or the average probability if producing a probabilistic result.

```python
C_list = [0.001, 0.005, 0.1, 0.5, 1, 10, 100, 1000, 10000]
class_weight = {0:.11, 1:.89}

logModel = LogisticRegressionCV(Cs=C_list, fit_intercept=True, penalty='l2', multi_class='ovr', 
                                class_weight=class_weight)
log_models = create_models(random.sample(detailed_train_playlists, 8), logModel)
```

The log_models are trained using the following function:

```python
# Given a model, fits the train data and returns it
def create_a_model(df, model):
    X_train = df.drop('hit', axis=1)
    y_train = df['hit']
    # standardize numerical values
    X_train = pd.DataFrame(X_train, columns=X_train.columns)

    model.fit(X_train, y_train)
    
    return model

# create a set of models to use for an ensembled result
def create_models(training_playlist_list, model):
    # need to make duplicates of the model or else the fitting will just happen to the same instance
    models = [copy.copy(model) for x in range(len(training_playlist_list))]
    # TODO: Try giving logistic model weights and run this for create_a_test_df instead
    final_models = [create_a_model(create_a_test_df(playlist)[0], model) \
                   for playlist, model in zip(training_playlist_list, models)]
    return final_models
```
The C-value for each model was cross-validated using LogisticRegressionCV. The number of models was determined by calculating the running accuracies for five different trees and visually selecting an N that performed optimally for all the playlists.

```python
# Create a logistic regression model
C_list = [0.001, 0.005, 0.1, 0.5, 1, 10, 100, 1000, 10000]
class_weight = {0:.11, 1:.89}
logModel = LogisticRegressionCV(Cs=C_list, fit_intercept=True, penalty='l2', multi_class='ovr', 
                                class_weight=class_weight)

# Check accuracy scores five times
for i in range(5):
    pred_log_matrix, y_log_train = predictions_matrix(logModel, 20)
    run_log_pred = running_predictions(pred_log_matrix, y_log_train.as_matrix())
    plt.scatter(range(len(run_log_pred)), run_log_pred, label='Run {}'.format(i))
plt.xlabel('Index')
plt.ylabel('Accuracy score')
plt.title('Number of logistic models vs accuracy score')
plt.legend()
```

The code to calculate the running predictions was slightly modified from the code provided by Harvard Univeristy, course CS-109A.

```python
def running_predictions(prediction_dataset, targets):
    n_models = prediction_dataset.shape[1]
    # find the running percentage of models voting 1 as more models are considered
    running_percent_1s = np.cumsum(prediction_dataset, axis=1)/np.arange(1,n_models+1)
    # predict 1 when the running average is above 0.5
    running_conclusions = running_percent_1s > 0.5
    # check whether the running predictions match the targets
    running_correctnesss = targets.reshape(-1,1) == running_conclusions
    # returns a 1-d series of the accuracy of using the first n trees to predict the targets
    return np.mean(running_correctnesss, axis=0)
```

The plot output for the running predictions indicates that accuracy is maximized at around 8 models.

![Logistic Regression Accuracy](/images/logistic_regression_accuracy.png)


<h3 id="2.3">2.3 Decision Tree</h3>
The decision tree model is a single tree of max depth 4. 0's were again given a class weight of .11 versus the 1's which were given a class weight of .89 to adjust for the large number of misses in the dataset versus the small number of hits. 

```python
DecisionTreeModel = DecisionTreeClassifier(max_depth=4)
DT_models = create_models(random.sample(detailed_train_playlists, 1), DecisionTreeModel)
```

The number of trees was selected by again calculating the running accuracies for five different trees and visually selecting an N that performed optimally for all the playlists.

```python
# run this 5 different times to cross validate results
class_weight = {0: .11, 1: .89}
DecisionTreeModel = DecisionTreeClassifier(max_depth=4, class_weight=class_weight)

for i in range(5):
    pred_tree_matrix, y_tree_train = predictions_matrix(DecisionTreeModel, 20)
    run_tree_pred = running_predictions(pred_tree_matrix, y_tree_train.as_matrix())
    plt.scatter(range(len(run_tree_pred)), run_tree_pred, label='Run {}'.format(i))
plt.xlabel('Index')
plt.ylabel('Accuracy score')
plt.title('Number of trees vs accuracy score')
plt.legend()
```

From the plot, accuracy does not change significantly when adding trees, so only a single tree is used for the model.

![Decision Tree Accuracy](/images/decision_tree_accuracy.png)

The decision tree is visualized below.

```python
from sklearn.tree import export_graphviz 
from sklearn.externals.six import StringIO  
import pydot 
from IPython.display import Image

dot_data = StringIO() 
export_graphviz(DT_models[0], dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
Image(graph[0].create_png())
```

![Decision Tree](/images/decision_tree.png)
<h3 id="2.4">2.4 Neural Network</h3>

Unlike the logistic or decision tree model, the neural network model is trained on 100 different dynamically generated datasets for 100 different `target song`s and `target playlist`s. This method is used instead of splitting the dataset into multiple batches, because the liklihood of creating batches of a good representation of hits and misses is low in such a skewed dataset. The classes are also given different weights with 0 being weighted .11 and 1 being weighted .89.

```python
# set input_size as number of predictors and num_class as 1 for binary
input_size = 6
num_classes = 1

NNmodel = Sequential([
    Dense(100, input_shape=(input_size,)),
    Activation('relu'),
    Dense(50),
    Activation('relu'),
    Dense(8),
    Activation('relu'),
    Dense(num_classes),
    Activation('sigmoid') # outputs probability between 0 and 1
])

NNmodel.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Instead of breaking up a single dataset into many batches, we will ask the neural net to train on the full
# dataset each time but fit it against multiple dynamic datasets
NN_train_dfs = [create_a_test_df(random.choice(detailed_train_playlists))[0] for i in range(100)]

for NN_train in NN_train_dfs:
    NN_X_train, NN_y_train = split_test_df(NN_train)
    num = Counter()
    num.update(NN_y_train)

    input_size = 6
    num_classes = 1
    # Classically in the case of strongly unbalanced classes,
    # the majority class is given a weight of .11 and the minority class a weight of .89
    class_weight = {0: .11, 1: .89}
        
    NNmodel.fit(NN_X_train, NN_y_train, epochs=1, batch_size=len(NN_X_train))
```

<h3 id="2.5">2.5 Ensemble Method</h3>
The metalearner model is an Adaboost model cross-validated for an optimal number of iterations. The three predictors are, given a `target track` and `target playlist`, the predictions of the probability of a hit from the three submodels: logistic regression model, decision tree model, and neural network model. Adaboost was chosen because we are able to tune the n_estimators to try and find the balance between over and underfitting. The final model is built below.

```python
final_train_predictions = build_train_predictions()
metaLearner = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators=20)
metaLearner.fit(final_train_predictions.drop('True', axis=1), final_train_predictions['True'])
```

Predictions resulting from the three models are combined into a single dataframe.

```python
# Selects a random track and playlist from the training set and builds a dataframe for it
def build_random_df(playlist_list):
    playlist = random.choice(playlist_list)
    df, target_song = create_a_test_df(playlist)
    X, y = split_test_df(df)
    return X, y

# Get a train prediction matrix from the 3 models
def build_train_predictions():
    X_train, y_train = build_random_df(detailed_train_playlists)
    
    # Logistic Model
    log_train_pred = get_predictions(X_train, y_train, log_models, prediction_type='predict_proba')

    # Decision Tree Model
    DT_train_pred = get_predictions(X_train, y_train, DT_models, prediction_type='predict_proba')

    # Neural Net Model
    NN_train_pred = [x[0] for x in NNmodel.predict(X_train)]

    # Aggregate the predictions of the log and decision tree models
    log_avg_train_pred = np.average(log_train_pred, axis=1)
    DT_avg_train_pred = np.average(DT_train_pred, axis=1)

    train_predictions = pd.DataFrame([log_avg_train_pred, DT_avg_train_pred, NN_train_pred, y_train]).transpose()
    train_predictions.columns=['Log', 'DT', 'NN', 'True']
    
    return train_predictions
```

An example of the resulting dataframe is displayed below:

![Metalearner DF](/images/metalearner_df.png)

To cross-validate for a reliable number of iterations, two different `target song`s and `target playlist`s were selected from the training set and used to train the AdaBoost metalearner. The running score was then plotted and analyzed for a shared optimum. The accuracy of the metalearner, instead of an overall accuracy, was the number of True Positives predicted by the model to the number of Total Positives.

```python
def build_running_adaboost(predictions):
    # Cross validate for a good n_estimators
    tp_to_pos_ratios = []
    fp_to_neg_ratios = []
    for i in range(1, 51):
        metaLearner = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators=i)
        metaLearner.fit(predictions.drop('True', axis=1), predictions['True'])

        tn, fp, fn, tp = confusion_matrix(metaLearner.predict(predictions.drop('True', axis=1)), 
                                          predictions['True']).transpose().ravel()
        
        tp_to_pos_ratios.append(tp/(tp+fn))

    return tp_to_pos_ratios

train_predictions = build_train_predictions()
cv_predictions = build_train_predictions()

train_tp_to_pos = build_running_adaboost(train_predictions)
cv_tp_to_pos = build_running_adaboost(cv_predictions)

plt.scatter(range(1, len(train_tp_to_pos)+1), train_tp_to_pos, label='Train')
plt.scatter(range(1, len(cv_tp_to_pos)+1), cv_tp_to_pos, label='CV')
plt.xlabel('Number of iterations')
plt.ylabel('Accuracy Score')
plt.title('Accuracy of Boosting Model vs Number of Iterations')
plt.legend()
```

![AdaBoost CV](/images/adaboost_cv.png)

From the plot, it appears that by 20 iterations, the model does well for both the train and cross-validation case, so n_estimators is set at 20 to prevent overfitting.

<h2 id="3">3. Last.FM Model</h2>
The LastFM Model will be an ensembled model of four sub-models. The four sub-models are as follows:
  - Logistic Regression
  - Bagging 
  - Boosting
  - Neural Network 

All four sub-models predict the probability of a song being a 'hit' (a song that appears on the target test playlist) and will be fed into a logistic regression model to combine the four predictions into a final class prediction. The result of the ensemble model will then feed into the final metal earner model - along with the output from the Last.FM model - to produce a final list of recommended songs.

<h3 id="3.1">3.1 Data Preparation</h3>
The input into the LastFM Model is a randomly selected song (called the `target track`) from a randomly selected playlist (called the `target playlist`). Given the `target track` and `target playlist`, a static `similars dataframe` of similar songs is generated by searching the LastFM `track database` (dataset) for the below features to build a dataframe with each row corresponding to a related song.

_Identification of relevant songs_
- Related songs by tag: any song that has at least one of the same tags associated with the target song will be added as a row; the number of tags in common with the target song will be stored as ‘tag_frequency’. This is a numerical variable.
- Related songs by artist: any song that is attributed to the same artist as the target song will be added as a row; the feature ‘same_artist’ will be added to the dataframe as a categorical variable; all of the songs found from this search will be given a ‘1’ since they were sung by the same artist; any song that was previously found by tag identification will be given a ‘1’ or ‘0’ for ‘same_artist’ as appropriate.

_Addition of more variables to the above identified songs_
- duration: numerical feature; relative duration will be calculated and included as a feature for each song; this will be relative to the target song: identified song duration / target song duration. 
- artist_hotness: numerical feature; relative artist hotness will be calculated and included as a feature for each song; this will be relative to the target song: (identified song artist hotness) / (target song artist hotness) 
- artist_familiarity: numerical feature; relative artist familiarity will be calculated and included as a feature; this will be relative to the target song:  (identified song artist familiarity) / (target song artist familiarity)
- same_album: categorical binary feature; any identified song that is from the same album as the target 

Once these pieces of the dataframe are built, a response variable of 0s and 1s will be added for each row: 1 if the identified song is in the playlist the target song was selected from and 0 otherwise. 

The code to generate one of these static `similars dataframe` is as follows:

```python
def get_target_track(playlist):
    temp_songs_only = [x.song for x in playlist]
    target_song = random.choice(list((set(temp_songs_only) & set(train_intersection_songs))))
    
    track_target = []
    for track in playlist:
        if target_song == track.song:
            track_target.append(track)
        else:
            pass
    return track_target[0]

def create_similars_dataframe(target_track, playlist, sorted_trackdf):
    #target_track = get_target_track(playlist)
    target_song, target_artist, target_album = (target_track.song, target_track.artist, target_track.album) 
    
    temp_title = sorted_trackdf['title'] == target_song
    temp_artist = sorted_trackdf['artist_name'] == target_artist
    temp_row = sorted_trackdf[temp_title & temp_artist]
    tags_needed = list(temp_row['tag_name'].values)
    tags_needed = [y for x in tags_needed for y in x]
    
    if len(tags_needed) == 0:
        dict_newdata_bytags = []
    else:    

        dict_newdata_bytags = []
        for i, row in enumerate(sorted_trackdf['tag_name']):
            if len(set(row) & set(tags_needed)) >= 2:
                new_data = {
                    'tag_frequency': len(set(tags_needed) & set(row)),
                    'song_title': sorted_trackdf['title'].iloc[i],
                    'same_album': int(sorted_trackdf['release'].iloc[i] == target_album),
                    'artist': sorted_trackdf['artist_name'].iloc[i],
                    'same_artist': int(sorted_trackdf['artist_name'].iloc[i] == target_artist),
                    'duration': (sorted_trackdf['duration'].iloc[i])/temp_row['duration'].values[0],
                    'artist_familiarity': (sorted_trackdf['artist_familiarity'].iloc[i])/temp_row[
                        'artist_familiarity'].values[0],
                    'artist_hotness': (sorted_trackdf['artist_hotness'].iloc[i])/temp_row['artist_hotness'].values[0]
                    }
                dict_newdata_bytags.append(new_data)
            else:
                pass
    
    
    temp_df = sorted_trackdf[sorted_trackdf['artist_name'] == target_artist]
    
    if len(temp_df) == 0:
        dict_newdata_byartist = []
    else:
        dict_newdata_byartist = []
        for i, row in enumerate(temp_df['artist_name']):
            new_data = {
                'tag_frequency': len(set(tags_needed) & set(temp_df['tag_name'].iloc[i])),
                'song_title': temp_df['title'].iloc[i],
                'same_album': int(temp_df['release'].iloc[i] == target_album),
                'artist': temp_df['artist_name'].iloc[i],
                'same_artist': 1,
                'duration': (temp_df['duration'].iloc[i])/temp_df['duration'].values[0],
                'artist_familiarity': (temp_df[
                    'artist_familiarity'].iloc[i])/temp_df['artist_familiarity'].values[0],
                'artist_hotness': (temp_df['artist_hotness'].iloc[i])/temp_df['artist_hotness'].values[0]
                }
            dict_newdata_byartist.append(new_data)

    if (dict_newdata_byartist == [] and dict_newdata_bytags == []):
        return None
    else: 
        bytag = pd.DataFrame(dict_newdata_bytags)
        byartist = pd.DataFrame(dict_newdata_byartist)

        full_df = pd.concat([bytag, byartist])
        full_df = full_df.drop_duplicates(subset = ['artist', 'song_title'], keep = 'first')

        # identify hits and misses for each similar song chosen
        hit = []
        temp_df_songs = list(full_df['song_title'].values)
        temp_songs_only = [x.song for x in playlist]
        same_songs = (set(temp_df_songs) & set(temp_songs_only))
        temp_df_artists = list(full_df['artist'].values)
        temp_artists_only = [x.artist for x in playlist]
        same_artist = (set(temp_df_artists) & set(temp_artists_only))
        for i, row in enumerate(full_df['song_title']):
            if (row in same_songs and full_df['artist'].iloc[i] in same_artist):
                hit.append(1)
            else:
                hit.append(0)
        full_df['hit'] = hit

        # re-index dataframe with song and artist info
        Track = namedtuple("Track", ["song", "artist"])
        new_index = []
        for i, row in enumerate(full_df['song_title']):
            new_index.append(Track(row, full_df['artist'].iloc[i]))
        full_df['track_info'] = new_index
        full_df.set_index('track_info', inplace=True)
        full_df = full_df.drop(['artist', 'song_title'], axis=1)
        del full_df.index.name
        
        return full_df
```

Here is an example of this executed for the `target song` 'Piano Man':
```python
target_track = detailed_train_playlists[playlists_piano[0]][18] # Piano Man target track
piano_man_df = create_similars_dataframe(target_track, 
                                         detailed_train_playlists[playlists_piano[0]], sorted_trackdf)
```
![PianoMan LastFM](/images/pianoman.png)

The process of creating the static `similars dataframe`s is repeated ~550 times (code below). Initially, the hope was to run it through 900 playlists, but after several hours, this required too much memory/processing power to handle. The output of repeating this ~550 times and concatenating into one large training dataframe leads to a dataset of over 20 million observations.

```python
df_list = []
for playlist in log_progress(final_train_lists): 
    target_track = get_target_track(playlist)
    temp_df = create_similars_dataframe(target_track, playlist, sorted_trackdf)
    df_list.append(temp_df)
final_train_df = pd.concat(df_list)
#log progress is a function appropriate source credit is in jupyter notebook, that allows you to track the progress of your for loops
```

Once the training dataframe has been created, the process is repeated with the test and tune playlists. 50 of the chosen test playlists and 75 of the tuning playlists are run through the process.

### Pre-processing
After the dataframes have all been created, some cleaning and pre-processing is done to ensure models can be fit to the data. Instances of infinity (explained further in jupyter notebook) were replaced with NaN, and all NaN observations were subsequently removed from the data. The dataframes were split into their features and response, and the numerical features were normalized:

```python
X_train = final_train_df.iloc[:, final_train_df.columns != 'hit']
y_train = final_train_df['hit'].values
X_test = final_test_df.iloc[:, final_train_df.columns != 'hit']
y_test = final_test_df['hit'].values
X_tune = final_tune_df.iloc[:, final_tune_df.columns != 'hit']
y_tune = final_tune_df['hit'].values

train_track_info = X_train.index.values
test_track_info = X_test.index.values
tune_track_info = X_tune.index.values

X_train = X_train.reset_index().drop('index', axis = 1) #indices needed to be reset for MinMaxScaler (returned error otherwise)
X_test = X_test.reset_index().drop('index', axis = 1)
X_tune = X_tune.reset_index().drop('index', axis = 1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(X_train[['artist_familiarity', 'artist_hotness', 'duration', 'tag_frequency']]) 
X_train[['artist_familiarity', 'artist_hotness', 'duration', 
         'tag_frequency']] = scaler.transform(X_train[['artist_familiarity', 'artist_hotness', 
                                                       'duration', 'tag_frequency']]) 
X_test[['artist_familiarity', 'artist_hotness', 'duration', 
         'tag_frequency']] = scaler.transform(X_test[['artist_familiarity', 'artist_hotness', 
                                                       'duration', 'tag_frequency']]) 
X_tune[['artist_familiarity', 'artist_hotness', 'duration', 
         'tag_frequency']] = scaler.transform(X_tune[['artist_familiarity', 'artist_hotness', 
                                                       'duration', 'tag_frequency']]) 
```
Now, the data are ready to be fit to models.

<h3 id="3.2">3.2 Logistic Regression</h3>
The logistic regression model was built with LogisticRegressionCV, with the default C-list and with class_weight = 'balanced'. Having a 'balanced' class weight is extremely important because 'misses' (class = 0) heavily dominate 'hits' (class = 1), and we want to 'even the playing field' between the two classes.

The logistic regression model was built with one line of code:
```python
logreg_model = LogisticRegressionCV(class_weight = 'balanced').fit(X_train, y_train)
```
Metrics and scores will be discussed in the Results section.

<h3 id="3.3">3.3 Bagging</h3>
The Bagging model was built with sklearn's BaggingClassifier, with the base estimator of a DecisionTreeClassifier. We plotted various depths of a DecisionTreeClassifier to determine the optimal depth for the BaggingClassifier:

```python
fig, ax = plt.subplots(1, 1, figsize = (15, 7))

list_scores = []
for cur_depth in range(1, 6):
    model = DecisionTreeClassifier(max_depth = cur_depth, class_weight = 'balanced')
    scores = cross_val_score(model, X_train, y_train, cv = 5)
    score_dict = {
        'depth': cur_depth,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'twostd_below': np.mean(scores) - np.std(scores)*2,
        'twostd_above': np.mean(scores) + np.std(scores)*2
    }
    list_scores.append(score_dict)
    ax.errorbar(x = cur_depth, y = np.mean(scores), yerr = np.std(scores)*2, fmt = 'o', lw = 3, 
                capsize = 6, markersize = 8)
    
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.tick_params(labelsize = 14)
ax.set_xlabel('Depth of Tree', fontsize = 15)
ax.set_ylabel('Performance of Model', fontsize = 15)
ax.set_title('Estimated Performance (+/- two s.d.)\nof Various Depths of a Decision Tree', fontsize = 18)
fig.subplots_adjust(hspace = .35, wspace = .25)

plt.show()
```

![dtree_plot](/images/decisiontree_plot.png)

The appropriate depth of the DecisionTree can be visualized below:

```python
DT_model = DecisionTreeClassifier(max_depth = 3, class_weight = 'balanced').fit(X_train, y_train)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(DT_model, out_file = dot_data,  
                filled = True, rounded = True,
                special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```

![dtree_nodes](/images/dtree_model.png)

Then, the ensembler model was fit with 25 estimators at a max_depth of 3. This number of estimators was chosen because that's the largest we felt we could bootstrap (with current memory/processing limitations) when there are > 20million observations in each boostrapped sample.

```python
bag_model = BaggingClassifier(
    base_estimator = DecisionTreeClassifier(max_depth = 3, class_weight = 'balanced'), 
    n_estimators = 25).fit(X_train, y_train)
```

<h3 id="3.4">3.4 Boosting</h3>
The boosting model was built with sklearn's AdaBoostClassifier. The first step we took to assess the classifier fit was plotting the classifier's scores on training and test sets at two depths and 15 estimators.

The two depths chosen were 2 and 3. Depth of 3, because that was the optimal depth for the BaggingClassifier and depth of 2, because AdaBoostClassifiers can often achieve optimal accuracy on lower depths than the base estimator (DecisionTree) can on its own. If there were more time and we had stronger processing power, we would have assessed the AdaBoost classifier on a broader range of depths and estimators (as we've done in class and homework).

```python
fig, axs = plt.subplots(1, 2, figsize = (15, 10))
axs = axs.ravel()

depths = [2, 3]

for i, cur_depth in log_progress(enumerate(depths), every = 1):
    model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = cur_depth), 
                               n_estimators = 15, learning_rate = 0.05).fit(X_train, y_train)
    train_staged_score = model.staged_score(X_train, y_train)
    test_staged_score = model.staged_score(X_test, y_test)

    train_scores = []
    for score in train_staged_score:
        train_scores.append(score)

    test_scores = []
    for score in test_staged_score:
        test_scores.append(score)
     
    stages = list(range(1, len(train_scores)+1))
    
    axs[i].plot(stages, train_scores, 'o', color = 'indigo', label = 'training')
    axs[i].plot(stages, test_scores, 'o', color = 'darkcyan', label = 'test')
    axs[i].set_xlabel('Number of Estimators', fontsize = 14)
    axs[i].set_ylabel('Accuracy Score', fontsize = 14)
    axs[i].set_title('Max Tree Depth of {}'.format(cur_depth), fontsize = 15)
    axs[i].legend(loc = 'best', fontsize = 15)
    axs[i].tick_params(labelsize = 13)
    #axs[i].set_ylim(0.960, 0.995)

fig.suptitle('Training and test scores as a function\nof the number of estimators in \
AdaBoost Classifier', fontsize = 20)
fig.tight_layout()
fig.subplots_adjust(top = 0.87, hspace = .35, wspace = .25)

plt.savefig('figures/AdaBoost.png')
plt.show()
```

![AdaBoost plot](/images/adaboost_lastfm.png)

On first glance, it looks like a terrible fit on both the training and the test sets. However, upon closer examination, we can see that the y_axis 'scale' is 9.998e-01, indicating that what looks like scores close to zero are actually scores close to 1. Since both depths 2 and 3 do quite well, we used a depth of 2 to train the classifier. We chose 4 estimators, as it's at that point on the plot that the training set in depth 2 first achieves its max score (and the test set had already reached its max).

```python
Ada_model = AdaBoostClassifier(
    base_estimator = DecisionTreeClassifier(max_depth = 2, class_weight = 'balanced'), 
    n_estimators = 4).fit(X_train, y_train)
```

<h3 id="3.5">3.5 Neural Network</h3>
The final sub-model of the LastFM models is a Neural Network. 

First, we converted the responses to a 2-column array of categorical responses to capture the 2 classes (0 and 1).

```python
from keras.utils import to_categorical

y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat  = to_categorical(y_test, num_classes=2)
```

Then, we chose to do a sequential model with rectified linear unit activation in every dense layer except for the output layer, for which we chose softmax.

```python
NN_model = Sequential([
    Dense(500, input_shape = (6,), activation='relu'),
    Dense(250, activation = 'relu'),
    Dense(100, activation = 'relu'),
    Dense(35, activation = 'relu'),
    Dense(15, activation = 'relu'),
    Dense(2, activation = 'softmax')
])

NN_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
NN_model.summary()
```

The compiled model had 157,957 total parameters. The model was run with class_weight: {0: 0.0005, 1:1.0}; the low number for the 0 class was an attempt to balance the millions of misses against the thousands of hits (class 1).

```python
class_weight = {0: 0.0005,
                1: 1.0}
NN_model.fit(X_train, y_train_cat, epochs = 1, batch_size = 32, 
             validation_split = 0.2, class_weight = class_weight)
```
The model ran for 40 minutes. While it had what seems like a great loss score (and accuracy score), the model performed very poorly. As we discuss in the results section, this is because it is accurately predicting all of the misses, but none of the hits.

<h3 id="3.6">3.6 Ensemble Model</h3>
To create the overall ensemble model of all of the LastFM submodels, we chose to do a logistic regression on the sub-models' predictions on fresh data (denoted `tune` in the following code).

```python
model_dict = {
    'logreg': logreg_model,
    'bagging': bag_model,
    'ada': Ada_model,
}

model_keys = model_dict.keys()
list_keys = list(model_keys)

ensemble_tune_array = np.zeros((len(X_tune), 3))
ensemble_test_array = np.zeros((len(X_test), 3))

for i, key in enumerate(model_keys):
    model = model_dict[key]
    ensemble_tune_array[:,i] = model.predict(X_tune)
    ensemble_test_array[:,i] = model.predict(X_test)
    
ensemble_tune = pd.DataFrame(ensemble_tune_array, columns = list_keys)
ensemble_test = pd.DataFrame(ensemble_test_array, columns = list_keys)

NN_tune = pd.DataFrame(NN_model.predict_classes(X_tune))
NN_tune = NN_tune.rename(columns = {NN_tune.columns[0]: 'NN'})
NN_test = pd.DataFrame(NN_model.predict_classes(X_test))
NN_test = NN_test.rename(columns = {NN_test.columns[0]: 'NN'})

ensemble_tune = pd.concat([ensemble_tune, NN_tune], axis = 1)
ensemble_test = pd.concat([ensemble_test, NN_test], axis = 1)

logreg_meta = LogisticRegressionCV(class_weight = 'balanced').fit(ensemble_tune, y_tune)
logreg_meta.coef_[0]
```

The ensemble model completely discarded the neural network, which makes sense given its poor performance. The logistic regression coefficients are shown below, with the neural net being the last:

![meta_coefs](/images/meta_coef_lastfm.png)


<h2 id="4">4. Metalearner</h2>

The final metalearner takes in the predictive output of both the Million Playlist model and the Last.FM model and combines them into a two-attribute dataframe. The final output is the probability of the recommended track being a hit.

Since both models are able to output a different set of tracks, when combined, any track missing due to one model recommending a track the other model does not are given a value of 0 for that model.

Logistic Regression was selected for the final metalearner because for both of the submodels, logistic regression seemed to perfrom well. 

![Two Model Prediction DF](/images/final_meta_df_v2.png)


Several functions were created to facilitate processing the predictions into a single dataframe.

```python
# concats two predictions and fills in NaNs with 0s
def two_models_one_result(prediction_1, prediction_2):
    result = pd.concat([prediction_1, prediction_2], axis = 1, sort = False)
    result.index = result.index.to_series()
    return result.fillna(value = 0)

# adds hit column by cross checking against the target playlist
def add_hit_column(result, target_playlist):
    target_playlist = strip_album(target_playlist)
    hit_list = []
    for idx in result.index:
        tempTrack = TrackLike(song=idx[0], artist=idx[1])
        if tempTrack in target_playlist:
            hit_list.append(1)
        else:
            hit_list.append(0)
            
    result['Hit'] = hit_list
    return result

# splits the DF into predictions and hit columns
def split_df(result):
    X = result.drop('Hit', axis=1)
    y = result['Hit']
    return X, y

# Strips album from the prediction index
def strip_album(playlist):
    newTrack = [TrackLike(song=playlist[i].song, artist=playlist[i].artist) \
              for i in range(len(playlist))]
    return newTrack
```

In order to train the model, we first randomly select a `target track` and `target playlist` from the training set. We then call the necessary functions to build the predictions.

```python
list_idx = random.sample(final_indices, 1)[0]
sample_MP_model_prediction = get_MP_model_prediction(detailed_train_playlists[list_idx][0], \
                                                     detailed_train_playlists[0])
prediction_1 = pd.DataFrame(sample_MP_model_prediction)
prediction_1 = strip_indices(prediction_1)

prediction_2 = get_lastfm_pred(get_target_track(detailed_train_playlists[list_idx]),
                               detailed_train_playlists[list_idx], sorted_trackdf)
```

After the submodels' predictions are built, the final metalearner is trained using the two predictions as attributes and the `target playlist` as the taret.

```python
final_train_df = add_hit_column(two_models_one_result(prediction_1, prediction_2), 
                                detailed_train_playlists[list_idx])
X_final_train, y_final_train = split_df(final_train_df)

C_list = [0.001, 0.005, 0.1, 0.5, 1, 10, 100, 1000, 10000, 100000, 1000000]
finalMetalearner = LogisticRegressionCV(Cs=C_list, fit_intercept=True, penalty='l2', multi_class='ovr', 
                                class_weight='balanced')
finalMetalearner.fit(X_final_train, y_final_train)
```

