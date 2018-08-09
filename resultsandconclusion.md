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
    * [3.1 Logistic Regression](#3.1)
    * [3.2 Decision Tree](#3.2)
    * [3.3 Neural Network](#3.3)
    * [3.4 Ensemble Method](#3.4)
* [4. Metalearner](#4)
* [5. Conclusion](#5)


<h2 id="1">1. Summary</h2>

Note:

All scores are mean values

TP = True Positive, FP = False Positive, FN = False Negative


<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>

<table class="tablelines">
  <thead>
    <tr>
      <th>Model</th>
      <th>Train Score</th>
      <th>Test Score</th>
      <th>Train True Positive/Total Positive ratio… Sensitivity</th>
      <th>Train False Positive/(True Positive + False Positive) ratio… FDR</th>
      <th>Train TP/(TP + FP)... Precision</th>
      <th>Test True Positive/Total Positive ratio</th>
       <th>Test False Positive/(True Positive + False Positive) ratio</th>
      <th>Parameters</th>
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
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td> </td>
      <td> </td>
      <td>MPD Ensemble Model</td>
      <td> 0.113055181696</td>
      <td> 0.00175202212554</td>
      <td> 0.998247977874</td>
      <td> 0.0865497076023</td>
      <td> 0.999084456259</td>
      <td> 0.000915543741074</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td>Neural Network</td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
      <td> </td>
    </tr>
  </tbody>
</table>


