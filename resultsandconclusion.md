---
layout: page
permalink: /resultsandconclusion/index.html
title: Results and Conclusions
---

# Results and Conclusions

## Contents

* [1. Summary](#1)
* [2. Million Playlist Model Results](#2)
* [3. Last.FM Model Results](#3)
* [4. Metalearner Results](#4)
* [5. Song Recommendations](#5)
* [6. Conclusion](#6)
* [7. Future considerations](#7)


<h2 id="1">1. Summary</h2>

Because of the large number of misses (encoded as 0) in our datasets, true accuracy scores are not very useful in understanding how well our models perform. We are trying to recommend a playlist based on a single song. Therefore, we primarily focused on three other metrics when determining our models' performance.

1. Sensitivity: Defined as `True Positive/(True Positive + False Negative)`. This value represents what we actually care about - out of all the tracks in the `target playlist`, how many did we actually recommend?

2. Precision: Defined as `True Positive/(True Positive + False Positive)`. This value represents how much our models are cheating - how many tracks are we recommending? If we recommend all the songs in our datasets, then we probably have a great sensitivity, but a recommendation this extensive wouldn't be very helpful.

3. False Discovery Rate (FDR): Defined as `False Positive/(True Positive + False Positive)`. This score is related to precision and tells us if our models are getting tracks correct simply because they are recommending everything and the kitchen sink. Note that the FDR is simply the complement of precision.

Below is an overview of how all our models performed. Each value is the mean score of either 100 train sets or 100 test sets.



<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1.5px solid black;
        }
</style>

<table class="tablelines" width="910">
  <thead>
    <tr align="center">
      <th width="130"><font size="3">Model</font></th>
      <th width="130">Sensitivity (Train)</th>
      <th width="130">Precision (Train)</th>
      <th width="130">False Discovery Rate (Train)</th>
      <th width="130">True Sensitivity (Test)</th>
      <th width="130">Sensitivity (Test)</th>
      <th width="130">Precision (Test)</th>
      <th width="130">False Discovery Rate (Test)</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center" bgcolor="#66B3FF">
      <td>Metalearner Model</td>
      <td> .1307</td>
      <td> .0199</td>
      <td> .9701</td>
      <td> .00004</td>
      <td> .1300</td>
      <td> .0064</td>
      <td> .9836</td>
    </tr>
    <tr align="center" bgcolor="#97CBFF">
      <td>Last.FM Ensemble</td>
      <td> 0.8813</td>
      <td> 0.0011</td>
      <td> 0.9989</td>
      <td> 0.0002</td>
      <td> 0.7345</td>
      <td> 0.0014</td>
      <td> 0.9986</td>
    </tr>
    <tr align="center" bgcolor="#C4E1FF">
      <td>Logistic Regression</td>
      <td> 0.7826</td>
      <td> 0.0014</td>
      <td> 0.9986</td>
      <td> 0.0001</td>
      <td> 0.7190</td>
      <td> 0.0014</td>
      <td> 0.9986</td>
    </tr>
    <tr align="center" bgcolor="#C4E1FF">
      <td>Bagging</td>
      <td> 0.7853</td>
      <td> 0.0014</td>
      <td> 0.9985</td>
      <td> 0.0001</td>
      <td> 0.6938</td>
      <td> 0.0013</td>
      <td> 0.9987</td>
    </tr>
    <tr align="center" bgcolor="#C4E1FF">
      <td>Boosting</td>
      <td> 0.8386</td>
      <td> 0.0013</td>
      <td> 0.9987</td>
      <td> 0.0002</td>
      <td> 0.7422</td>
      <td> 0.0001</td>
      <td> 0.9989</td>
    </tr>
    <tr align="center" bgcolor="#C4E1FF">
      <td>Neural Network</td>
      <td> 0.0</td>
      <td> 0.0</td>
      <td> 0.0</td>
      <td> 0.0</td>
      <td> 0.0</td>
      <td> 0.0</td>
      <td> 0.0</td>
    </tr>
    <tr align="center" bgcolor="#97CBFF">
      <td> MPD Ensemble Model</td>
      <td> 0.1131</td>
      <td> 0.0018</td>
      <td> 0.9982</td>
      <td> 0.0865</td>
      <td> 0.0865</td>
      <td> 0.0009</td>
      <td> 0.9991</td>
    </tr>
    <tr align="center" bgcolor="#C4E1FF">
      <td>Logistic Regression</td>
      <td> 0.1412</td>
      <td> 0.0390</td>
      <td> 0.9610</td>
      <td> 0.1193</td>
      <td> 0.1586</td>
      <td> 0.0240</td>
      <td> 0.9760</td>
    </tr>
    <tr align="center" bgcolor="#C4E1FF">
      <td>Decision Tree</td>
      <td> 0.0125</td>
      <td> 0.0474</td>
      <td> 0.9526</td>
      <td> 0.0200</td>
      <td> 0.0150</td>
      <td> 0.0355</td>
      <td> 0.9645</td>
    </tr>
    <tr align="center" bgcolor="#C4E1FF">
      <td>Neural Network</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
      <td> 0</td>
    </tr>
  </tbody>
</table>


<h2 id="2">2. Million Playlist Model Results</h2>

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


<h2 id="3">3. Last.FM Model Results</h2>

The code below was run with each model after it was fit to return a table of metrics. This table included: sensitivity, true sensitivity, precision, false discovery rate, specificity, and accuracy. True sensitiviy is a metric we created to assess how many hits the model returned in relation to all of the songs returned. It is calculated by dividing the true positive value by the number of tracks in the dataframe.

```python
def metrics_models(y_true, y_pred, col_name):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    metrics_dict = {
        'sensitivity': TP/(TP + FN),
        'precision': TP/(TP + FP),
        'fdr': FP/(TP + FP), #false discovery rate (complement of precision)
        'specificity': TN/(TN + FP),
        'accuracy': (TP + TN)/(TN + FP + FN + TP),
    }
    return pd.DataFrame.from_dict(metrics_dict, orient = 'index', columns = [col_name])

logreg_train_metrics = metrics_models(y_train, logreg_model.predict(X_train), 'logreg_train')
logreg_test_metrics = metrics_models(y_test, logreg_model.predict(X_test), 'logreg_test')
logreg_metrics = pd.concat([logreg_train_metrics, logreg_test_metrics], axis = 1)

DT_train_metrics = metrics_models(y_train, DT_model.predict(X_train), 'tree_train')
DT_test_metrics = metrics_models(y_test, DT_model.predict(X_test), 'tree_test')
DT_metrics = pd.concat([DT_train_metrics, DT_test_metrics], axis = 1)

bag_train_metrics = metrics_models(y_train, bag_model.predict(X_train), 'bagging_train')
bag_test_metrics = metrics_models(y_test, bag_model.predict(X_test), 'bagging_test')
bag_metrics = pd.concat([bag_train_metrics, bag_test_metrics], axis = 1)

Ada_train_metrics = metrics_models(y_train, Ada_model.predict(X_train), 'Ada_train')
Ada_test_metrics = metrics_models(y_test, Ada_model.predict(X_test), 'Ada_test')
Ada_metrics = pd.concat([Ada_train_metrics, Ada_test_metrics], axis = 1)

# Neural network required different code; the model performed quite poorly and return 0s in the 'false positive' and 'true positive' parts of the confusion matrix; with a return of 0, some of the metrics were technically infinity by the actual calculations; we've defaulted the infinity values to 0
NN_train_metrics_dict = {
    'sensitivity': 0,
    'true_sensitivity': 0,
    'precision': 0,
    'fdr': 0, 
    'specificity': 1,
    'accuracy': (nn_tp_train + nn_tn_train)/(nn_tn_train + nn_fp_train + nn_fn_train + nn_tp_train),
    }

NN_test_metrics_dict = {
    'sensitivity': 0,
    'true_sensitivity': 0,
    'precision': 0,
    'fdr': 0, 
    'specificity': 1,
    'accuracy': (nn_tp_test + nn_tn_test)/(nn_tn_test + nn_fp_test + nn_fn_test + nn_tp_test),
    }

NN_metrics = pd.DataFrame.from_dict([NN_train_metrics_dict, NN_test_metrics_dict])
NN_metrics = NN_metrics[['sensitivity', 'true_sensitivity', 'precision', 'fdr', 
                         'specificity', 'accuracy']]
NN_metrics = NN_metrics.transpose()
NN_metrics = NN_metrics.rename(index=str, columns={0: "NN_train", 1: "NN_test"})

meta_tune_metrics = metrics_models(y_tune, logreg_meta.predict(ensemble_tune), 'meta_tune')
meta_test_metrics = metrics_models(y_test, logreg_meta.predict(ensemble_test), 'meta_test')
meta_metrics = pd.concat([meta_tune_metrics, meta_test_metrics], axis = 1)

full_test_metrics = pd.concat([logreg_test_metrics, bag_test_metrics, Ada_test_metrics, 
                               NN_metrics['NN_test'], meta_test_metrics], axis = 1)
                               
full_train_metrics = pd.concat([logreg_train_metrics, bag_train_metrics, Ada_train_metrics, 
                                NN_metrics['NN_train'], meta_tune_metrics], axis = 1)
```

### Summary of results

All of the models (submodels and the ensembler model), have relatively high specificity and accuracy. The models do well with finding true negatives and the majority of the `similars dataframe` contains true negatives (or true misses). Because of this, both specificity and accuracy are high. 

The neural network performed the worst of all four submodels. Logistic regression, bagging, and boosting showed relatively high sensitivity (~70%); however, they all scored extremely low (~0.01%) on the true sensitivity metric. The ensembler model built out of all four models showed a similar pattern. Our understanding of this is that the model is relatively good at detecting hits out of the songs it has included in the `similars dataframe` (sensitivity), however the majority of the `similars dataframe` includes songs that are not found in the `target playlist` (true sensitivity).

All of the models and the ensembler performed poorly on precision (~0.1%, except for neural nets, with 0%); this demonstrates that the models predicted many more false positives than true positives. Since the purpose of this model is to recommend songs a user might like, based off of a `target song` in a `target playlist`, it is important that the model has higher precision and recommends fewer false positives. False discovery rate is the complement of precision, so it follows that all models scored extremely high (~99%) on this metric. The neural network scored 0% in both precision and false discovery network because we assigned 0s to any metric with a 0 in its denominator to avoid a divison by zero error. Otherwise, these metrics could be viewed as Not Applicable.

Below are the detailed metrics for the models, organized by training and testing.

#### Training Metrics Table

<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1.5px solid black;
        }
</style>

<table class="tablelines" width="910">
  <thead>
    <tr align="center">
      <th width="130"><font size="3"></font></th>
      <th width="130">Sensitivity</th>
      <th width="130">True Sensitivity</th>
      <th width="130">Precision</th>
      <th width="130">False discovery rate</th>
      <th width="130">Specificity</th>
      <th width="130">Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>Logistic regression</td>
      <td> 0.7826</td>
      <td> 0.0002</td>
      <td> 0.0014</td>
      <td> 0.9986</td>
      <td> 0.8829</td>
      <td> 0.8829</td>
    </tr>
    <tr align="center">
      <td>Bagging</td>
      <td> 0.7853</td>
      <td> 0.0002</td>
      <td> 0.0015</td>
      <td> 0.9985</td>
      <td> 0.8918</td>
      <td> 0.8918</td>
    </tr>
     <tr align="center">
      <td>Boosting</td>
      <td> 0.8386</td>
      <td> 0.0002</td>
      <td> 0.0013</td>
      <td> 0.9987</td>
      <td> 0.8677</td>
      <td> 0.8677</td>
    </tr>
    <tr align="center">
      <td>Neural network</td>
      <td> 0.0000</td>
      <td> 0.0000</td>
      <td> 0.0000</td>
      <td> 0.0000</td>
      <td> 1.0000</td>
      <td> 0.9998</td>
    </tr>
    <tr align="center">
      <td>Ensemble model</td>
      <td> 0.8813</td>
      <td> 0.0002</td>
      <td> 0.0011</td>
      <td> 0.9989</td>
      <td> 0.8521</td>
      <td> 0.8522</td>
   </tr>
  </tbody>
</table>


#### Test Metrics Table

<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1.5px solid black;
        }
</style>

<table class="tablelines" width="910">
  <thead>
    <tr align="center">
      <th width="130"> </th>
      <th width="130">Sensitivity</th>
      <th width="130">True Sensitivity</th>
      <th width="130">Precision</th>
      <th width="130">False discovery rate</th>
      <th width="130">Specificity</th>
      <th width="130">Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>Logistic regression</td>
      <td> 0.7190</td>
      <td> 0.0001</td>
      <td> 0.0014</td>
      <td> 0.9986</td>
      <td> 0.8938</td>
      <td> 0.8937</td>
    </tr>
    <tr align="center">
      <td>Bagging</td>
      <td> 0.6938</td>
      <td> 0.0001</td>
      <td> 0.0013</td>
      <td> 0.9987</td>
      <td> 0.8915</td>
      <td> 0.8915</td>
    </tr>
     <tr align="center">
      <td>Boosting</td>
      <td> 0.7422</td>
      <td> 0.0002</td>
      <td> 0.0012</td>
      <td> 0.9988</td>
      <td> 0.8760</td>
      <td> 0.8760</td>
    </tr>
    <tr align="center">
      <td>Neural network</td>
      <td> 0.0000</td>
      <td> 0.0000</td>
      <td> 0.0000</td>
      <td> 0.0000</td>
      <td> 1.0000</td>
      <td> 0.9998</td>
    </tr>
    <tr align="center">
      <td>Ensemble model</td>
      <td> 0.7345</td>
      <td> 0.0002</td>
      <td> 0.0014</td>
      <td> 0.9986</td>
      <td> 0.8886</td>
      <td> 0.8885</td>
   </tr>
  </tbody>
</table>



<h2 id="4">4. Metalearner Results</h2>
Once the meta learner was trained using the two predictions as features and the `target playlist` as the response, it was tested on a subset of the training playlists and the test playlists that were used to test the ensemble models. 

In order to this, we used the following functions to report metrics on the metalearner (and its submodels):

```python
def run_100_final_trains():
    train_tn_list = []
    train_fp_list = []
    train_fn_list = []
    train_tp_list = []
    train_tot_list = []
    
    for idx in random.sample(final_indices, 100):
        target_track = get_target_track(detailed_train_playlists[idx])
        MP_model_prediction = pd.DataFrame(get_MP_model_prediction(target_track, 
                                                                   detailed_train_playlists[idx]))
        MP_model_prediction = strip_indices(MP_model_prediction)
        FM_model_prediction = get_lastfm_pred(target_track, detailed_train_playlists[idx], sorted_trackdf)
        final_train_df = two_models_one_result(MP_model_prediction, FM_model_prediction)
        final_train_df = add_hit_column(final_train_df, detailed_train_playlists[idx])

        X_final_train, y_final_train = split_df(final_train_df)
        
        if X_final_train.shape[1] < 2:
            X_final_train['LastFM'] = 0.0
        else:
            X_final_train = X_final_train
        
        predicted_rec = finalMetalearner.predict(X_final_train)
        
        train_tn, train_fp, train_fn, train_tp = \
            confusion_matrix(y_final_train, predicted_rec).ravel() 
        train_tn_list.append(train_tn)
        train_fp_list.append(train_fp)
        train_fn_list.append(train_fn)
        train_tp_list.append(train_tp) 
        train_tot_list.append(len(X_final_train))
        
    return train_tn_list, train_fp_list, train_fn_list, train_tp_list, train_tot_list

def run_100_final_tests():
    test_tn_list = []
    test_fp_list = []
    test_fn_list = []
    test_tp_list = []
    test_tot_list = []
    
    mp_test_tn_list = []
    mp_test_fp_list = []
    mp_test_fn_list = []
    mp_test_tp_list = []
    mp_test_tot_list = []
    
    fm_test_tn_list = []
    fm_test_fp_list = []
    fm_test_fn_list = []
    fm_test_tp_list = []
    fm_test_tot_list = []
    
    for idx in test_idx_songs:
        test_playlist = detailed_test_playlists[idx['playlist_idx']]
        target_track = detailed_test_playlists[idx['playlist_idx']][idx['song_idx']]
        
        MP_model_prediction = pd.DataFrame(get_MP_model_prediction(target_track, test_playlist))
        MP_model_prediction = strip_indices(MP_model_prediction)
        FM_model_prediction = get_lastfm_pred(target_track, test_playlist, sorted_trackdf)
       
        final_test_df = two_models_one_result(MP_model_prediction, FM_model_prediction)
        final_test_df = add_hit_column(final_test_df, test_playlist)
        X_test, y_test = split_df(final_test_df)
                
        if X_test.shape[1] < 2:
            X_test['LastFM'] = 0.0
        else:
            X_test = X_test
            
        # Turn results binary to get submodel confusion matrices
        binary_predictions = pd.DataFrame([X_test['Meta_Prob'].apply(lambda x: 0 if x <= .5 else 1),
                                           X_test['LastFM'].apply(lambda x: 0 if x <= .5 else 1)]).transpose()
        
        # Record confusion matrix stats for MP model only
        try:
            mp_test_tn, mp_test_fp, mp_test_fn, mp_test_tp = confusion_matrix(y_test, binary_predictions['Meta_Prob']).ravel()
        except ValueError:
            mp_test_tn = confusion_matrix(y_test, binary_predictions['Meta_Prob']).ravel()[0]
            mp_test_fp, mp_test_fn, mp_test_tp = (0,0,0)
        
        mp_test_tn_list.append(mp_test_tn)
        mp_test_fp_list.append(mp_test_fp)
        mp_test_fn_list.append(mp_test_fn)
        mp_test_tp_list.append(mp_test_tp)
        mp_test_tot_list.append(len(X_test))
        
        # Record confusion matrix for FM model only
        try:
            fm_test_tn, fm_test_fp, fm_test_fn, fm_test_tp = confusion_matrix(y_test, binary_predictions['LastFM']).ravel()
        except ValueError:
            fm_test_tn = confusion_matrix(y_test, binary_predictions['LastFM']).ravel()[0]
            fm_test_fp, fm_test_fn, fm_test_tp = (0,0,0)
        
        fm_test_tn_list.append(fm_test_tn)
        fm_test_fp_list.append(fm_test_fp)
        fm_test_fn_list.append(fm_test_fn)
        fm_test_tp_list.append(fm_test_tp)
        fm_test_tot_list.append(len(X_test))
            
        
        predicted_rec = finalMetalearner.predict(X_test)        
                
        try:
            test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, predicted_rec).ravel()
        except ValueError:
            test_tn = confusion_matrix(y_test, predicted_rec).ravel()[0]
            test_fp, test_fn, test_tp = (0,0,0)
        test_tn_list.append(test_tn)
        test_fp_list.append(test_fp)
        test_fn_list.append(test_fn)
        test_tp_list.append(test_tp)
        test_tot_list.append(len(X_test))
    return test_tn_list, test_fp_list, test_fn_list, test_tp_list, test_tot_list, \
            mp_test_tn_list, mp_test_fp_list, mp_test_fn_list, mp_test_tp_list, \
            mp_test_tot_list, fm_test_tn_list, fm_test_fp_list, fm_test_fn_list, \
            fm_test_tp_list, fm_test_tot_list
```

From these functions, we were able to obtain the following:

For the train playlists:
- on the meta learner: true negative, false positive, false negative, true positive, and total length of the features 

```python
train_tn_list, train_fp_list, train_fn_list, train_tp_list, train_tot_list = run_100_final_trains()
```

For the test playlists:
- on the meta learner: true negative, false positive, false negative, true positive, and total length of the features
- on the LastFM model: true negative, false positive, false negative, true positive, and total length of the features
- on the Million Playlist Dataset model: true negative, false positive, false negative, true positive, and total length of the features  

```python
test_tn_list, test_fp_list, test_fn_list, test_tp_list, test_tot_list, \
mp_test_tn_list, mp_test_fp_list, mp_test_fn_list, mp_test_tp_list, mp_test_tot_list, \
fm_test_tn_list, fm_test_fp_list, fm_test_fn_list, fm_test_tp_list, fm_test_tot_list = run_100_final_tests()
```
While we ran 100 train sets and recorded the overall sensitivity, precision, and discovery scores in the above results table, we used a slightly different function against the 100 test sets. We captured the true negative, false positive, false negative, and true positive values for the two submodels as well as the metalearner model. We used these intermediary statistics to compare the performance of each of the submodels versus the final metalearner. Below is the visualization of sensitivity, precision, and true sensitivity scores of the three models' predictions using test data.

```python
labels = ['Million Playlist Model', 'Last.FM Model', 'Metalearner Model']
sensitivity_labels = ['Sensitivity', 'Precision', 'True Sensitivity']

meta_graph = [meta_sensitivity_list, meta_precision_list, meta_true_sensitivity_list]
fm_graph = [fm_sensitivity_list, fm_precision_list, fm_true_sensitivity_list]
mp_graph = [mp_sensitivity_list, mp_precision_list, mp_true_sensitivity_list]

fig, ax = plt.subplots(nrows=3, ncols=1)
fig.set_size_inches(15, 20)
fig.suptitle('Comparison of Sensitivity and Precision Scores for Submodels and Metalearner Model', 
             fontsize=20, y=0.95)

for i in range(3):
    ax[i].boxplot([mp_graph[i], fm_graph[i], meta_graph[i]], 
                   labels=labels)
    ax[i].legend()
    ax[i].set_ylabel('Score')
    ax[i].set_title('Comparison of {} Scores'.format(sensitivity_labels[i]))
    #ax[i].set_ylim(ymax=.4)
```

![Final Results](/images/final_results.png)

From these visualizations, we observe that all three models (Million Playlist Model, LastFM Model, and Metalearner Model) all have very low true sensitivity scores; LastFM model is the worst performer on this metric, with Million Playlist Model and Metalearner Model having comparable performances (similar median value). Based on the metrics performed on the two submodels during their own ensembling process, this finding makes sense. The LastFM submodels and the ensemble model had extremely low true sensitivity (0.0002), while the Million Playlist ensemble model and its submodels consistently performed better (with scores ranging from 0.01 to 0.09).

The models performed similarly in precision, as all models have a median very close to zero. LastFM’s 3rd quartile (and consequently, its maximum) is slightly higher than the Million Playlist Model and the Metalearner model. 

Of interest, and what demonstrates the complexities of combining two models that were built with different types of data frames, LastFM’s sensitivity is much lower than both Million Playlist Model and metalearner Model. However, the LastFM model outperforms the Million Playlist Model in sensitivity (~0.73 vs ~0.11, respectively) during the process of initially generating these models and their submodels. One explanation behind this difference in sensitivity behavior could be how the LastFM model was built; this model was generated using one static, large, stacked data frame. The process of creating the metalearner introduces only one playlist at a time to the models, which is a change to how LastFM was trained and built. 

<h2 id="5">5. Song recommendation</h2>
To achieve our ultimate goal of providing a user with recommended songs based on a `target track` in one of their current playlists, we performed the following, which is a modification of `run_100_final_train()`, applied to a test playlist:

```python
test_idx = random.sample(final_indices, 1)[0]
target_track = get_target_track(detailed_train_playlists[test_idx])
MP_model_prediction = pd.DataFrame(get_MP_model_prediction(target_track, 
                                                                   detailed_train_playlists[test_idx]))
MP_model_prediction = strip_indices(MP_model_prediction)
FM_model_prediction = get_lastfm_pred(target_track, detailed_train_playlists[test_idx], sorted_trackdf)
final_train_df = two_models_one_result(MP_model_prediction, FM_model_prediction)
final_train_df = add_hit_column(final_train_df, detailed_train_playlists[test_idx])

X_final_train, y_final_train = split_df(final_train_df)
predicted_rec = finalMetalearner.predict(X_final_train)
predicted_rec_df = pd.DataFrame([predicted_rec, y_final_train]).transpose().set_index(X_final_train.index)
predicted_rec_df.groupby(0).get_group(1)
        
train_tn, train_fp, train_fn, train_tp = \
            confusion_matrix(y_final_train, predicted_rec).ravel() 
```

The resulting data frame from predicted_rec_df.groupby(0).get_group(1) provides a list of recommendations through the track name and artist in the indices. The ones in the ‘0’ column indicate that the model predicted the song as a hit. The 0s and 1s in the ‘1’ column indicate whether or not that song was actually found in the `target playlist`. So, for Led Zeppelin’s ‘Whole Lotta Love’, the model recommends 6919 tracks, including: (Anna Sun, WALK THE MOON), (I Bet You Look Good On The Dancefloor, Arctic Monkeys), (From The Ritz To The Rubble, Arctic Monkeys), (Trojans, Atlas Genius), (Two Against One (feat. Jack White), Danger Mouse).

![song_recs](/images/song_recs.png)

We only performed this process once, as it would require a lot of memory and time to predict songs for 100 train playlists and 100 test playlists. However, for a given playlist, the below function will provide a set of song recommendations and associated metrics with the predictions:

```python
def get_song_pred(target_playlist, sorted_trackdf)
target_track = get_target_track(target_playlist)
MP_model_prediction = pd.DataFrame(get_MP_model_prediction(target_track, 
                                                                   target_playlist))
MP_model_prediction = strip_indices(MP_model_prediction)
FM_model_prediction = get_lastfm_pred(target_track, target_playlist, sorted_trackdf)
final_train_df = two_models_one_result(MP_model_prediction, FM_model_prediction)
final_train_df = add_hit_column(final_train_df, target_playlist)

X_final_train, y_final_train = split_df(final_train_df)
predicted_rec = finalMetalearner.predict(X_final_train)
predicted_rec_df = pd.DataFrame([predicted_rec, y_final_train]).transpose().set_index(X_final_train.index)
rec_songs = predicted_rec_df.groupby(0).get_group(1).index
        
tot_list = len(X_final_train)

train_tn, train_fp, train_fn, train_tp = \
            confusion_matrix(y_final_train, predicted_rec).ravel() 

return rec_songs, train_tn, train_fp, train_fn, train_tp, tot_list
```

<h2 id="6">6. Conclusion</h2>

In this study, we explore a difficult recommendation problem where, given just a single song, we recommend a list of songs we believe the user will like. Our model will be most helpful as the algorithm for a MRS when the MRS has no other information about a user other than their first song selection. After all, even for these new users, we want to provide a recommendation that is better than a random guess.

To achieve this goal, we applied many different models and variations of those models to uniquely generated datasets. We used pre-existing playlists to validate our recommendations under the assumption that, if our model can recommend other songs from the playlist given one of the songs from the playlist, then our model is doing a good job.

We showcased a problem that had many categorical features that would require one-hot encoding (artist, album, tags, etc.) and limited those features by tailoring datasets to a specific `target song`. Our results show that, while our features may not have been the best features to use, they were still able to produce sucessful predictions.

We learned that individual models perform better or worse than others under certain metrics, and that applying ensembling methods to several models in some cases will and in other cases won't improve certain metrics. Choosing "the best" model is not always straight forward, especially when dealing with imbalanced data. However, in our case, it seemed like the simpler models (logistic regressions) generally performed better than the more complex models (neural networks).

There were many challenges related to the data that forced us to make decisions on our recommendation algorithm: large amounts of input data, imbalanced data, dynamic data, and vastly different datasets. At the crossroads of each of these decisions, it would have been optimal to cross validate several options before moving forward. However, given time constraints, it became clear that we would have to make intuitive decisions (or ensemble enough models to make decisions for us) in order to move forward.

<h2 id="7">7. Future considerations</h2>

If given more time we would like to invest in the following:

1. Allocating enough memory to load all playlist data at once.
2. Tuning and cross-validating the parameters of all sub-models and ensemblers.
3. Trying different class weights and other methods of dealing with imbalanced data.
4. Exploring other methods of building the dynamic track list.
5. Using additional datasets like the Million Song Dataset with audio features.
