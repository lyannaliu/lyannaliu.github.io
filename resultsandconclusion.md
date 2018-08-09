---
layout: page
permalink: /resultsandconclusion/index.html
title: Results and Conclusions
---

# Results and Conclusions

* [1. Summary](#1)
* [2. Last.FM Model Results](#2)
    * [2.1 Logistic regression](#2.1)
    * [2.2 Ensemble model](#2.2)
* [3. Million Playlist Model Results](#3)
   
* [5. Conclusion](#5)


<h2 id="1">1. Summary</h2>





<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>

<table class="tablelines">
  <thead>
    <tr>
      <th>Model</th>
      <th>Sensitivity (Train)</th>
      <th>Precision (Train)</th>
      <th>False Discovery Rate (Train)</th>
      <th>Sensitivity (Test)</th>
      <th>Precision (Test)</th>
      <th>False Discovery Rate (Test)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Metalearner Model</td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td>Last.FM Ensemble</td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td>Other model 2</td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td>Other model 3</td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td> MPD Ensemble Model</td>
      <td> 0.113055181696</td>
      <td> 0.00175202212554</td>
      <td> 0.998247977874</td>
      <td> 0.0865497076023</td>
      <td> 0.000915543741074</td>
      <td> 0.999084456259</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td> 0.141184387618</td>
      <td> 0.0389600742804</td>
      <td> 0.96103992572</td>
      <td> 0.119298245614</td>
      <td> 0.0239838763856</td>
      <td> 0.976016123614</td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td> 0.0125168236878</td>
      <td> 0.0474247832738</td>
      <td> 0.952575216726</td>
      <td> 0.015037593985</td>
      <td> 0.0355169692186</td>
      <td> 0.964483030781</td>
    </tr>
    <tr>
      <td>Neural Network</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
    </tr>
  </tbody>
</table>

<h2 id="2">2. Last.FM Model Results</h2>
Last.FM Model Results
<h3 id="2.1">2.1 Logistic regression</h3>
Logistic regression
<h3 id="2.2">2.2 Ensemble model</h3>
Ensemble model
<h2 id="3">3. Million Playlist Model Results</h2>

Several custom functions were built to automate the selection of test `target track`s and `target playlist`s, building predictions on the selection, and processing the results.

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

Then, the results were averaged to calculate the overall sensitivity, precision, false discovery rate, and true sensitivity of the Million Playlist Model against test sets. These values are reported in the results table above.

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

# Calculate total average stats
log_test_sensitivity, log_test_precision, log_test_fdr, log_test_true_sensitivity= average_test_results(log_test_res)
DT_test_sensitivity, DT_test_precision, DT_test_fdr, DT_test_true_sensitivity = average_test_results(DT_test_res)
NN_test_sensitivity, NN_test_precision, NN_test_fdr, NN_test_true_sensitivity = average_test_results(NN_test_res)
meta_test_sensitivity, meta_test_precision, meta_test_fdr, meta_test_true_sensitvity = average_test_results(meta_test_res)
```

From the calculations, we observe that the logistic regression model has the highest average sensitivity. The decision tree model has the highest average precision. The ensemble model does slightly worse than the best model in all respects. The neural network was unable to predict any hits.

We visually compared the results of the sensitivity scores of each of the models to understand more about their predictive powers.

```python
labels = ['Log', 'DT', 'NN', 'Meta']
sensitivity_labels = ['Train', 'Test', 'Calculated True Sensitivity from Test']

log_sensitivity_graph = [log_sensitivity_train_list, log_sensitivity_test_list, log_true_sensitivity_test_list]
DT_sensitivity_graph = [DT_sensitivity_train_list, DT_sensitivity_test_list, DT_true_sensitivity_test_list]
NN_sensitivity_graph = [NN_sensitivity_train_list, NN_sensitivity_test_list, NN_true_sensitivity_test_list]
meta_sensitivity_graph = [meta_sensitivity_train_list, meta_sensitivity_test_list, meta_true_sensitivity_test_list]


fig, ax = plt.subplots(nrows=3, ncols=1)
fig.set_size_inches(15, 20)
fig.suptitle('Comparison of Sensitivity Scores of Train and Test Predictions Across All Models', 
             fontsize=20, y=0.95)

for count_type, count_type_name, i in zip(count_types, count_type_names, range(3)):
    ax[i].boxplot([log_sensitivity_graph[i], DT_sensitivity_graph[i], NN_sensitivity_graph[i], 
                   meta_sensitivity_graph[i]], labels=labels)
    ax[i].legend()
    ax[i].set_ylabel('Sensitivity Score')
    ax[i].set_xlabel('Index')
    ax[i].set_title('{} Cases'.format(sensitivity_labels[i]))
    
```

![Sensitivity Score Comparison All Models](/images/comparison_of_sensitivity_scores_all_models.png)

It appears that, although the logistic regression model is able to predict with the highest level of sensitivity, the median sensitivity score of the metalearner is actually higher. The standard deviation of the scores is also notably smaller. The metalearner also has fewer scores closer to 0 than any of the other models. Additionally, the scores for the train and test sets are overall very similar, which means we most likely avoided overfitting the training sets.

We then visually compared the results of the precision scores of each of the models.

```python
labels = ['Log', 'DT', 'NN', 'Meta']
precision_labels = ['Train', 'Test']

log_precision_graph = [log_precision_train_list, log_precision_test_list]
DT_precision_graph = [DT_precision_train_list, DT_precision_test_list]
NN_precision_graph = [NN_precision_train_list, NN_precision_test_list]
meta_precision_graph = [meta_precision_train_list, meta_sensitivity_test_list]


fig, ax = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(15, 20)
fig.suptitle('Comparison of Precision Scores of Train and Test Predictions Across All Models', 
             fontsize=20, y=0.95)

for count_type, count_type_name, i in zip(count_types, count_type_names, range(2)):
    ax[i].boxplot([log_precision_graph[i], DT_precision_graph[i], NN_precision_graph[i], 
                   meta_precision_graph[i]], labels=labels)
    ax[i].legend()
    ax[i].set_ylabel('Precision Score')
    ax[i].set_xlabel('Index')
    ax[i].set_title('{} Cases'.format(precision_labels[i]))
```

![Precision Score Comparison All Models](/images/comparison_of_precision_scores_all_models.png)

Interestingly, all models had around the same low levels of precision against the training set with the logistic regression model performing the best, but the metalearner actually had much higher median and overall precision in the test set although the overal mean of the scores is lower than the logistic regression model. This may be due to the logistic model performing extremely well in a significant number of cases.

Lastly, we visualy compared the results of the false discovery rates of each of the models.

```python
labels = ['Log', 'DT', 'NN', 'Meta']
fdr_labels = ['Train', 'Test']

log_fdr_graph = [log_fdr_train_list, log_fdr_test_list]
DT_fdr_graph = [DT_fdr_train_list, DT_fdr_test_list]
NN_fdr_graph = [NN_fdr_train_list, NN_fdr_test_list]
meta_fdr_graph = [meta_fdr_train_list, meta_fdr_test_list]


fig, ax = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(15, 20)
fig.suptitle('Comparison of False Discovery Scores of Train and Test Predictions Across All Models', 
             fontsize=20, y=0.95)

for count_type, count_type_name, i in zip(count_types, count_type_names, range(2)):
    ax[i].boxplot([log_fdr_graph[i], DT_fdr_graph[i], NN_fdr_graph[i], 
                   meta_fdr_graph[i]], labels=labels)
    ax[i].legend()
    ax[i].set_ylabel('False Discovery Score')
    ax[i].set_xlabel('Index')
    ax[i].set_title('{} Cases'.format(fdr_labels[i]))
```

![FDR Score Comparison All Models](/images/comparison_of_fdr_scores_all_models.png)

The decision tree had an enormous spread of false discover rates from 0 to 1, but also had the lowest median false discover rate. The metamodel had the highest median but the tightest standard deviation. The logistic model only performed slightly better than the metalearner model.

Overall, out of the three submodels, logistic regression performed the best. In general, it was also able to predict more false positives than even the metalearner model. However, the metalearner had generally the tightest spreads in its prediction scores. It's possible that the metalearner was hurt by the Neural Network model which seemed to significantly overfit for the 0 scores, and the fact that the decision tree model and neural network models were not preferred in any of the metrics.


<h2 id="5">5 Conclusion</h2>
Conclusion
