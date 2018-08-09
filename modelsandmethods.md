---
layout: page
permalink: /modelsandmethods/index.html
title: Models and Methods
---

# Models and Methods

#### Contents

* [1. Overlook of the Models](#1)
* [2. Last.FM Model](#2)
    * [2.1 Data Preparation](#2.1)
    * [2.2 Logistic regression](#2.2)
    * [2.3 Ensemble model](#2.3)
* [3. Million Playlist Model](#3)
    * [3.1 Data Preparation](#3.1)
    * [3.2 Logistic Regression](#3.2)
    * [3.3 Decision Tree](#3.3)
    * [3.4 Neural Network](#3.4)
    * [3.5 Ensemble Method](#3.5)
* [4. Metalearner](#4)

<h2 id="1">1. Overlook of the Models</h2>
The overall structure of the model description of metamodel and two submodels:

![Fig](/images/Fig.png)

<h2 id="2">2. Last.FM Model</h2>
The Last FM Model was created as an ensemble model, built from three other models: 

  - Bagging 
  - Boosting
  - Neural Network 
  
**Need to be completed:** 

All three models will predict the probability of a song being a 'hit': the predicted song appears on the  

<h3 id="2.1">2.1 Data Preparation</h3>
building dynamic dfs
Include snapshot of one dynamic df
<h3 id="2.2">2.2 Logistic regression</h3>
Code to create models
Affiliated diagrams
Justification
<h3 id="2.3">2.3 Ensemble model</h3>
Code to ensemble above models
Affiliated diagrams
ustification


<h2 id="3">3. Million Playlist Model</h2>
The Million Playlist Model will be an ensembled model of three sub-models and a metalearner. The first sub-model is a logistic regression model, the second is a decision tree, and the third is a neural net. All three sub-models predict the probability of a song being a 'hit' (a song that appears on the target test playlist) and be fed into an Adaboost metalearner model that will combine the three predictions into a final probabilistic prediction. The result of the Adaboost metalearner will then feed into the final ensembler model - along with the output from the Last.FM model - to produce a final list of recommended songs.
<h3 id="3.1">3.1 Data Preparation</h3>
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

<h3 id="3.2">3.2 Logistic Regression</h3>
The logistic regression submodel is an ensembled set of eight logistic models. The eight models were trained separately on eight randomly selected `target tracks` and `target playlists` from the detailed_train_playlists set. 0's were given a class weight of .11 versus the 1's which were given a class weight of .89 to adjust for the large number of misses in the dataset versus the small number of hits. The final prediction is either the majority if producing a binary result or the average probability if producing a probabilistic result.

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

The code below generates the prediction matrix for the logistic regression models.

```python
# Create predictions matrix for N log models
def predictions_matrix(model, N=20):
    # Train a large number (20) of training models
    train_models = create_models(random.sample(detailed_train_playlists, N), model)
    
    # create a test tf using one of the train playlists
    train_df, train_song = create_a_test_df(random.choice(detailed_train_playlists))
    X_train, y_train = split_test_df(train_df)
    
    predictions = get_predictions(X_train, y_train, train_models)

    return predictions, y_train
```

The plot output for the running predictions indicates that accuracy is maximized at around 8 models.

![Logistic Regression Accuracy](/images/logistic_regression_accuracy.png)


<h3 id="3.3">3.3 Decision Tree</h3>
The decision tree model is a single tree of max depth 4. 0's were given a class weight of .11 versus the 1's which were given a class weight of .89 to adjust for the large number of misses in the dataset versus the small number of hits. 

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
<h3 id="3.4">3.4 Neural Network</h3>

Unlike the logistic or decision tree model, the neural network model is trained on 100 different dynamically generated datasets for 100 different `target song`s and `target playlist`s. This method is used instead of splitting the dataset into multiple batches, because the liklihood of creating batches of a good representation of hits and misses is low in such a skewed dataset. The classes are given different weights with 0 being weighted .11 and 1 being weighted .89.

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

<h3 id="3.5">3.5 Ensemble Method</h3>
The metalearner model is an Adaboost model cross validated for an optimal number of iterations. The three predictors are, given a `target track` and `target playlist`, the predictions of the probability of a hit from the three submodels: logistic regression model, decision tree model, and neural network model. Adaboost was chosen because we are able to tune the n_estimators to try and find the balance between over and underfitting. The final model is built below.

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

To crossvalidate for a reliable number of iterations, two different `target song`s and `target playlist`s were selected from the training set and used to train the AdaBoost metalearner. The running score was then plotted and analyzed for a shared optimum. The accuracy of the metalearner, instead of an overall accuracy, was the number of True Positives predicted by the model to the number of Total Positives.

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

From the plot, it appears that by 20 iterations, the model does well for both the train and cross validation case, so n_estimators is set at 20 to prevent overfitting.

<h2 id="4">4. Metalearner</h2>
Description
Challenges
Model
Code
Justification: Why we thought this was a good way to build the model

