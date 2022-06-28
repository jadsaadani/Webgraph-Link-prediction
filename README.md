# Link-prediction
Machine Learning project about link prediction for the French web.
Team: Oualid EL HAJOUJI, Othman GAIZI, Jad SAADANI

# Introduction

The aim of this work is to design a Machine Learning algorithm that predicts the presence of
a link between pages of a subgraph of the French webgraph. More precisely, we delete a
certain number of existing links in the subgraph, and then attempt to predict the existence of
a link between two given vertices (pages). Throughout this work, we mainly focus on feature
engineering and selection. Indeed, the given input consists in the modified oriented subgraph
(basically an adjacency matrix) and the html text associated to each node of the network: this
relatively raw data has to be processed into new actionable features before creating any
classification model. In order to create new features, we spotted two approaches. The first
one consists in relying on the graph structure for feature creation: the features associated to
a couple of nodes would be graph measures such as the degree or the Katz centrality. The
second one is based on the html text: NLP methods can be used in order to leverage this data
and make it useful for our prediction. Once the features created, we test different classifier
models and select the relevant features, for each classifier, among the created ones. The
metric that assesses the quality of our models is the Mean F1-Score. In fact, a baseline model
that relies solely on the Jaccard Coefficient of the pairs of nodes achieves a score of 0.64.

Link to the InClass Kaggle competition: https://www.kaggle.com/c/link-prediction-data-challenge-2019/overview

# Files

cleaning_textdata.py contains the code that cleans the text data.
link-prediction-main.py produces the prediction on the test set.
