# Deep Learning Project 2 - PUBG winning prediction

This is one of the Artificial Neural Network course projects. In the project, we implemented various CNNs (ResNet, SEREsNet, VGG, DenseNet, REsNeXt) to predict the wining rate using [PUBG dataset](https://www.kaggle.com/c/pubg-finish-placement-prediction). (For the privacy sake I delete all our real names here.)

## Project Abstract

PLAYERUNKNOWN'S BATTLEGROUNDS is drawing more and more attention in recent years, whose matching dataset is available in Kaggle for contestants worldwide to explore.  However, only a few neural network model is applied to the prediction of the player's win percentage, and no convolution neural network (CNN) is witnessed in top-ranked submissions. In this paper, we implement a series of networks, including basic MLP and various CNN with fine-tuning. Then, we analyze their performances and conclude their drawbacks in PUBG dataset. After walking through other high-ranked submissions, we then visualize the dataset and figure out the inner relation between them by feature engineering. Finally, we implement GDBT and random forest (RF) regression with extra features we built in feature engineering. Our results show not only the shortcomings of CNN in handling dataset with strong and sparse relation but also the overwhelming advantages of feature engineering. The performances of classic machine learning models then prove our hypothesis.

## File Structure

- `report.pdf` the final report of the whole project,
- `read_data.py` the PUBG dataset reader and sampler,
- `network-ipynb/` the notebook of the CNN that we implemented, including ResNet, SEREsNet, VGG, DenseNet, REsNeXt,
- `network-pdf/` the pdf version of `network-ipynb/`,
- `machine-learning/` other machine learning methods, including GBDT and RF, to predict the winning rate.