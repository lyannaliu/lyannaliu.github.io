---
layout: page
permalink: /descriptionofdata/index.html
title: Description of Data
---

# Description of Data

* [1. Summary](#1)
* [2. LastFM](#2)
    * [2.1 Data Source](#2.1)
    * [2.2 Description of the Raw Data](#2.2)
    * [2.3 Exploratory Data Analysis](#2.3)
* [3. Million Playlist](#3)
    * [3.1 Data Source](#3.1)
    * [3.2 Description of the Raw Data](#3.2)
    * [3.3 Exploratory Data Analysis](#3.3)

<h2 id="1">1. Summary</h2>
We used these two data sets

<h2 id="2">2. LastFM</h2>
Last.FM
<h3 id="2.1">2.1 Data Source</h3>
Where did it come from?
<h3 id="2.2">2.2 Description of the Raw Data</h3>
What is contained in the data?
Include df to illustrate
Issues with the data?
<h4 id="2.3">2.3 Exploratory Data Analysis</h4>
Code used to process/clean data
Code used to create visualizations
Code used to compare with Million Playlist data

<h2 id="3">3. Million Playlist</h2>
Million Playlist
<h3 id="3.1">3.1 Data Source</h3>
Where did it come from?
Code to read in files
<h3 id="3.2">3.2 Description of the Raw Data</h3>
What is contained in the data?
Include one dictionary/df to illustrate - Haley
Issues with the data?
Steps taken to clean
Code used to clean data
<h4 id="3.3">3.3 Exploratory Data Analysis</h4>
The first step to analyzing the Million Playlist Data was to decide the data structure that would best suit the needs of both the Last.FM Model and Million Playlist Model. The biggest issue with the playlist data was ensuring that tracks could be identified as unique while keeping the amount of loaded information to a minimum (to reduce memory costs).

There were several instances of tracks with the same song name, and several instances of tracks with the same song and artist but different albums. Therefore, we made the following decisions:

- Remastered/Remixes of the same song performed by the same artists are considered different tracks.
- The same song performed by different artists are considered different tracks.
- The same songs performed by the same artist in different albums are considered different tracks.

The most obvious way to store a track was as a `namedtuple` with three properties - song, artist, and album.
```python
Track = namedtuple("Track", ["song", "artist", "album"])
```

Using this template, we loaded each playlist as a list of Track `namedtuple`s, and the entire list of playlists as a list of lists.

```python
def randomly_load_files():
    all_playlists = []
    all_files = glob.glob("mpd.v1/data/100000/*.json")
    for file in all_files:
        with open(file) as f:
            data = json.load(f)
            all_playlists.extend([[Track(song=track['track_name'], artist=track['artist_name'], 
                album=track['album_name']) for track in playlist['tracks']] for playlist in data['playlists']])
    return all_playlists
```

Another challenge we faced was memory restrictions. Loading the full list of 1,000,000 playlists wasn't feasible, so we trimmed the data down to just 100,000 playlists. Out of these 100,000 playlists, 90% were designated train and 10% were designated test. After loading the data, we achieved this random split using `sklearn`'s `train_test_split` method.

```python
all_playlists = randomly_load_files()
detailed_train_playlists, detailed_test_playlists = train_test_split(all_playlists, train_size=.9)
```

The training and test sets were pickled and used for both the Last.FM and Million Playlist models. This handled the issue of creating train and test sets of data.

```python
with open('detailed_train_playlists.pkl', 'wb') as f:
    pickle.dump(detailed_train_playlists, f)

with open('detailed_test_playlists.pkl', 'wb') as f:
    pickle.dump(detailed_test_playlists, f)
```

The next question we needed to answer was how to turn completely categorical data with too many variables to 1-hot encode into preferably numerical data that could be ingested by a model.

We began by exploring the concept of calculating total counts of songs, artists, and albums from the training set as potential attributes for a track.

```python
# get counts of all unique songs, artists, and albums
def get_unique(playlist_list):
    totalArtists = Counter()
    totalAlbums = Counter()
    details = Counter()
    for playlist in playlist_list:
        for track in playlist:
            totalArtists[track.artist] += 1
            totalAlbums[track.album] += 1
            details[track] += 1
    return (totalArtists, totalAlbums, details)

totalArtistCount, totalAlbumCount, songDetails = get_unique(detailed_train_playlists)
```

The above function returns three `Counter`s (that function similarly to a dictionary) that contain the number of times each unique track, artist, and album appears across all training playlists.

```python
# Count of tracks
songDetails.most_common()
[(Track(song='HUMBLE.', artist='Kendrick Lamar', album='DAMN.'), 3984),
 (Track(song='One Dance', artist='Drake', album='Views'), 3844),
 (Track(song='Closer', artist='The Chainsmokers', album='Closer'), 3730),
 (Track(song='Broccoli (feat. Lil Yachty)', artist='DRAM', album='Big Baby DRAM'),
  3679),
 (Track(song='Congratulations', artist='Post Malone', album='Stoney'), 3538),
 (Track(song='Caroline', artist='Aminé', album='Good For You'), 3172),
 (Track(song='iSpy (feat. Lil Yachty)', artist='KYLE', album='iSpy (feat. Lil Yachty)'),
  3135),
 (Track(song='Location', artist='Khalid', album='American Teen'), 3103),
 (Track(song='XO TOUR Llif3', artist='Lil Uzi Vert', album='Luv Is Rage 2'),
  3100),
 (Track(song='Bad and Boujee (feat. Lil Uzi Vert)', artist='Migos', album='Culture'),
  3051),
 (Track(song='No Role Modelz', artist='J. Cole', album='2014 Forest Hills Drive'),
  2899),
 (Track(song='Bounce Back', artist='Big Sean', album='I Decided.'), 2860),
 (Track(song='Ignition - Remix', artist='R. Kelly', album='Chocolate Factory'),
  2860),
  ...

# Count of artists
totalArtistCount.most_common()

[('Drake', 74492),
 ('Kanye West', 37094),
 ('Kendrick Lamar', 31078),
 ('Rihanna', 30070),
 ('The Weeknd', 28218),
 ('Eminem', 25941),
 ('Ed Sheeran', 24748),
 ('Future', 22572),
 ('J. Cole', 21718),
 ('Justin Bieber', 21235),
 ('Beyoncé', 21235),
 ('The Chainsmokers', 19982),
 ('Chris Brown', 18625),
 ('Luke Bryan', 18580),
 ('Twenty One Pilots', 17924),
 ('Calvin Harris', 17851),
 ('Lil Uzi Vert', 17612),
 ('Post Malone', 17240),
 ...

 # Count of albums
 [('Views', 18436),
 ('Stoney', 13837),
 ('Greatest Hits', 13504),
 ('More Life', 12396),
 ('DAMN.', 12341),
 ('Beauty Behind The Madness', 12287),
 ('Coloring Book', 11868),
 ('American Teen', 10906),
 ('Culture', 10712),
 ('The Life Of Pablo', 10214),
 ('Purpose', 10114),
 ('2014 Forest Hills Drive', 9766),
 ('Starboy', 9566),
 ('Blurryface', 9488),
 ('ANTI', 9350),
 ('÷', 9190),
 ('Original Album Classics', 9185),
 ('x', 8885),
 ('Montevallo', 8815)
 ...
 ```

 Next, we were interested in visualizing any trends for these counts by plotting the calculated frequencies.

```python
# Plot total track, artist, and album counts
count_types = [songDetails, totalArtistCount, totalAlbumCount]
count_type_names = ['Tracks', 'Artists', 'Albums']

fig, ax = plt.subplots(nrows=3, ncols=1)
fig.set_size_inches(10, 15)
fig.suptitle('Total Counts of Tracks, Artists, and Albums', fontsize=20, y=0.95)

for count_type, count_type_name, i in zip(count_types, count_type_names, range(3)):
    ax[i].scatter(range(len(count_type)), count_type.values(), label=count_type_name)
    ax[i].set_xlabel('Indexes')
    ax[i].set_ylabel('Total count')
    ax[i].set_title('Total counts of {}'.format(count_type_name))
```

![Total Counts](total_counts.png)

By doing so we discovered that all three follow a similar Pareto distribution. Colinearity became a concern, so we selected ten tracks and compared the trend of the total song count, artist count, and album counts of each song.

```python
# Plot subset of counts
from matplotlib.lines import Line2D

colors = ['blueviolet', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'darkorchid', 'fuchsia',
         'indianred', 'mediumblue', 'lightpink']
markers = [Line2D([], [], marker='.', markersize=10, label='Song Count'),
           Line2D([], [], marker='x', markersize=10, label='Artist Count'), 
           Line2D([], [], marker='o', markersize=10, label='Album Count')]

fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
fig.suptitle('Total Counts of a Subset of Tracks, Artists, and Albums', fontsize=20, y=0.95)

for i in range(10):
    ax.scatter(i, songDetails[detailed_train_playlists[0][i]], marker='.', color=colors[i])
    ax.scatter(i, totalArtistCount[detailed_train_playlists[0][i].artist], marker='x', color=colors[i])
    ax.scatter(i, totalAlbumCount[detailed_train_playlists[0][i].album], marker='o', color=colors[i])

ax.set_xlabel('Indexes')
ax.set_ylabel('Total count')
ax.legend(handles=markers)
```

![Count Comparison](count_comparison.png)

The plot shows that songs with high counts do not necessarily have artists and albums with high counts. The same holds true for the relationship between artists and albums to songs. Therefore, even though there is some visible colinearity, we determined that the frequency counts were not perfectly colinear and could potentially be significant as separate predictors.

# Million Playlist Model

The Million Playlist Model will be an ensembled model of three sub-models and a metalearner. The first sub-model is a logistic regression model, the second is a decision tree, and the third is a neural net. All three sub-models predict the probability of a song being a 'hit' (a song that appears on the target test playlist) and be fed into an Adaboost metalearner model that will combine the three predictions into a final probabilistic prediction. The result of the Adaboost metalearner will then feed into the final ensembler model - along with the output from the Last.FM model - to produce a final list of recommended songs.

# Data Preparation

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

![Piano Man DF](piano_man_df.png)

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

![Piano Man DF](piano_man_df.png)

# Logistic Regression Submodel

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

![Logistic Regression Accuracy](logistic_regression_accuracy.png)


# Decision Tree Model

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

![Decision Tree Accuracy](decision_tree_accuracy.png)

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

![Decision Tree](decision_tree.png)

# Neural Network Model

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

# Metalearner Model

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

![Metalearner DF](metalearner_df.png)

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

![AdaBoost CV](adaboost_cv.png)

From the plot, it appears that by 20 iterations, the model does well for both the train and cross validation case, so n_estimators is set at 20 to prevent overfitting.

# Results


# Million Playlist Results

INSERT TABLE HERE

Several custom functions were built to automate selecting test `target track`s and `target playlist`s, building predictions on the selection, and processing the results.

```python
def build_random_test_df():
    test_playlist = random.choice(detailed_test_playlists)
    test_df, target_song = create_a_test_df(test_playlist)
    X_test, y_test = split_test_df(test_df)
    return X_test, y_test
    
# Get a train prediction matrix from the 3 models
def build_test_predictions(X_test, y_test):
    # Logistic Model
    log_pred = get_predictions(X_test, y_test, log_models, prediction_type='predict_proba')

    # Decision Tree Model
    DT_pred = get_predictions(X_test, y_test, DT_models, prediction_type='predict_proba')

    # Neural Net Model
    NN_pred = [x[0] for x in NNmodel.predict(X_test)]

    # Aggregate the predictions of the log and decision tree models
    log_avg_pred = np.average(log_pred, axis=1)
    DT_avg_pred = np.average(DT_pred, axis=1)

    test_predictions = pd.DataFrame([log_avg_pred, DT_avg_pred, NN_pred, y_test]).transpose().set_index(X_test.index)
    test_predictions.columns=['Log', 'DT', 'NN', 'True']
    
    return test_predictions

# Given a submodel prediction matrix, returns the metalearner's binary prediction results
def get_metalearner_binary_prediction(predictions):
    # Cross validate for a good n_estimators

    metalearner_binary_predictions = metaLearner.predict(predictions.drop('True', axis=1))
    final_binary_predictions = pd.DataFrame([metalearner_binary_predictions, predictions['True']])\
        .transpose().set_index(predictions.index)
    final_binary_predictions.columns=['Meta_Binary', 'True']

    tn, fp, fn, tp = confusion_matrix(final_binary_predictions['Meta_Binary'], 
                                      final_binary_predictions['True']).transpose().ravel()

    return final_binary_predictions, tp/(tp+fn), fp/(fp+tn)

# Given a submodel prediction matrix, returns the metalearner's probabilistic prediction results
def get_metalearner_prob_prediction(predictions):
    # probability of class 1
    metalearner_prob_predictions = metaLearner.predict_proba(predictions.drop('True', axis=1))[:,1]
    final_prob_predictions = pd.DataFrame([metalearner_prob_predictions, predictions['True']])\
        .transpose().set_index(predictions.index)
    final_prob_predictions.columns=['Meta_Prob', 'True']
    
    return final_prob_predictions

# Converts submodel's predictions from probabilistic to binary by an adjustable decision boundary
def get_binary_submodel_predictions(prob_predictions):
    # Convert to 0 and 1
    # TODO: At what % convert to 0 and 1?
    binary_predictions = pd.DataFrame([prob_predictions['Log'].apply(lambda x: 0 if x <= .5 else 1),
                                            prob_predictions['DT'].apply(lambda x: 0 if x <= .5 else 1),
                                            prob_predictions['NN'].apply(lambda x: 0 if x <= .5 else 1),
                                            prob_predictions['True']]).transpose()
    return binary_predictions

# Converts metalearner's predictions from probabilistic to binary by an adjustable decision boundary
def get_binary_metalearner_predictions(prob_predictions):
    binary_predictions = pd.DataFrame([prob_predictions['Meta_Prob'].apply(lambda x: 0 if x <= .5 else 1),
                                            prob_predictions['True']]).transpose()
    return binary_predictions
```

Using these functions, analysis could be performed on the true negative, false positive, false negative, and true positives from the predictive results.

First, the true negative, false positive, false negative, and true positives were calculated for 100 different train samples.

```python
# Predicts on 100 training playlists and returns confusion matrix results
def train_100_results():
    log_res = []
    DT_res = []
    NN_res = []
    meta_res = []
    
    for i in range(100):
        train_prediction = build_train_predictions()
        bin_prediction = get_binary_submodel_predictions(train_prediction)
        log_tn, log_fp, log_fn, log_tp = confusion_matrix(bin_prediction['Log'], 
                                      bin_prediction['True']).transpose().ravel()
        log_res.append([log_tn, log_fp, log_fn, log_tp])
        
        DT_tn, DT_fp, DT_fn, DT_tp = confusion_matrix(bin_prediction['DT'], 
                                      bin_prediction['True']).transpose().ravel()
        DT_res.append([DT_tn, DT_fp, DT_fn, DT_tp])
        
        NN_tn, NN_fp, NN_fn, NN_tp = confusion_matrix(bin_prediction['NN'], 
                                      bin_prediction['True']).transpose().ravel()
        NN_res.append([NN_tn, NN_fp, NN_fn, NN_tp])
        
        bin_meta_prediction, _, _= get_metalearner_binary_prediction(train_prediction)
        meta_tn, meta_fp, meta_fn, meta_tp = confusion_matrix(bin_meta_prediction['Meta_Binary'], 
                                      bin_meta_prediction['True']).transpose().ravel()
        meta_res.append([meta_tn, meta_fp, meta_fn, meta_tp])

    return log_res, DT_res, NN_res, meta_res

log_train_res, DT_train_res, NN_train_res, meta_train_res = train_100_results()


# Calculates mean results of a list of confusion matrix stuff
def average_results(confusion_matrix_res):
    tn = sum([res[0] for res in confusion_matrix_res])
    fp = sum([res[1] for res in confusion_matrix_res])
    fn = sum([res[2] for res in confusion_matrix_res])
    tp = sum([res[3] for res in confusion_matrix_res])
    
    sensitivity = tp/(tp + fn)
    precision = tp/(tp + fp)
    fdr = fp/(tp+fp)
    
    return sensitivity, precision, fdr
```

Then, the results were averaged to calculate the overall sensitivity, precision, and false discovery rate of the Million Playlist Model against train sets.

```python
# Average train results
log_train_sensitivity, log_train_precision, log_train_fdr = average_results(log_train_res)
DT_train_sensitivity, DT_train_precision, DT_train_fdr = average_results(DT_train_res)
NN_train_sensitivity, NN_train_precision, NN_train_fdr = average_results(NN_train_res)
meta_train_sensitivity, meta_train_precision, meta_train_fdr = average_results(meta_train_res)
```

Similarly, the true negative, false positive, false negative, and true positives were calculated for 100 different test samples. However, unlike for the train samples, there is a possibility that a `target track` from the test set is not found in the set of detailed_training_playlists. In this case, no related tracks will be found, and the dynamic dataframe will only consist of several randomly selected songs, and the models will be likely to predict 0 for these songs. To accurately score the model, two sensitivity scores are recorded - True Positive/(True Positive + False Negative) and True Positive/Total Length of `Target Playlist`. In most datasets, the two would be equal; however, since our dynamically created playlists can realistically build dataframes without all of the songs from the `target playlist`, this is not necessarily the case for our Million Playlist Model. The latter calculation is what is reported in the main results table.

```python
# Randomly selected list of test playlists and test songs by index
pkl_file = open('idx_title_test_playlist.pkl', 'rb')
test_idx_songs = pickle.load(pkl_file)

# Predicts on test results and returns confusion matrix results
def test_100_results():
    log_res = []
    DT_res = []
    NN_res = []
    meta_res = []
    
    for idx in test_idx_songs:
        playlist = detailed_test_playlists[idx['playlist_idx']]
        playlist_len = len(playlist)
        df, _ = create_a_test_df(playlist, detailed_test_playlists[idx['playlist_idx']][idx['song_idx']])
        X, y = split_test_df(df)
        test_pred = build_test_predictions(X, y)
        
        bin_prediction = get_binary_submodel_predictions(test_pred)
        
        try:
            log_tn, log_fp, log_fn, log_tp = confusion_matrix(bin_prediction['Log'], 
                                          bin_prediction['True']).transpose().ravel()
            log_res.append([log_tn, log_fp, log_fn, log_tp, playlist_len])
        except ValueError:
            log_tn = confusion_matrix(bin_prediction['Log'], 
                                          bin_prediction['True']).ravel()[0]
            log_res.append([log_tn, 0, 0, 0, playlist_len])

        try:
            DT_tn, DT_fp, DT_fn, DT_tp = confusion_matrix(bin_prediction['DT'], 
                                          bin_prediction['True']).transpose().ravel()
            DT_res.append([DT_tn, DT_fp, DT_fn, DT_tp, playlist_len])
        except ValueError:
            
            DT_tn = confusion_matrix(bin_prediction['DT'], 
                                          bin_prediction['True']).ravel()[0]
            DT_res.append([DT_tn, 0, 0, 0, playlist_len])

        try:
            NN_tn, NN_fp, NN_fn, NN_tp = confusion_matrix(bin_prediction['NN'], 
                                          bin_prediction['True']).transpose().ravel()
            NN_res.append([NN_tn, NN_fp, NN_fn, NN_tp, playlist_len])
        except ValueError:
            NN_tn = confusion_matrix(bin_prediction['NN'], 
                                          bin_prediction['True']).ravel()[0]
            NN_res.append([NN_tn, 0, 0, 0, playlist_len])
        
        bin_meta_prediction = get_metalearner_binary_prediction(test_pred, metrics=False)
            
        try:
            meta_tn, meta_fp, meta_fn, meta_tp = confusion_matrix(bin_meta_prediction['Meta_Binary'], 
                                          bin_meta_prediction['True']).transpose().ravel()
            meta_res.append([meta_tn, meta_fp, meta_fn, meta_tp, playlist_len])
        except ValueError:
            print(idx)
            meta_tn = confusion_matrix(bin_meta_prediction['Meta_Binary'], 
                                          bin_meta_prediction['True']).ravel()[0]
            meta_res.append([meta_tn, 0, 0, 0, playlist_len])
            
            
    return log_res, DT_res, NN_res, meta_res

log_test_res, DT_test_res, NN_test_res, meta_test_res = test_100_results()
```

Then, the results were averaged to calculate the overall sensitivity, precision, false discovery rate, and true sensitivity of the Million Playlist Model against test sets.

```python
# Calculates the average performance of the test set
def average_test_results(confusion_matrix_res):
    tn = sum([res[0] for res in confusion_matrix_res])
    fp = sum([res[1] for res in confusion_matrix_res])
    fn = sum([res[2] for res in confusion_matrix_res])
    tp = sum([res[3] for res in confusion_matrix_res])
    tot = sum([res[4] for res in confusion_matrix_res])
    
    sensitivity = tp/(tp + fn)
    precision = tp/(tp + fp)
    fdr = fp/(tp+fp)
    true_sensitivity = tp/tot
    
    return sensitivity, precision, fdr, true_sensitivity

# MOST IMPORTANT RESULTS TO BE ADDED TO A TABLE
log_test_sensitivity, log_test_precision, log_test_fdr, log_test_true_sensitivity= average_test_results(log_test_res)
DT_test_sensitivity, DT_test_precision, DT_test_fdr, DT_test_true_sensitivity = average_test_results(DT_test_res)
NN_test_sensitivity, NN_test_precision, NN_test_fdr, NN_test_true_sensitivity = average_test_results(NN_test_res)
meta_test_sensitivity, meta_test_precision, meta_test_fdr, meta_test_true_sensitvity = average_test_results(meta_test_res)
```


