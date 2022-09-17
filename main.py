####################################
# Dror Manzur
# ID: 301317798
####################################
# Part A: rows: 30-430
# Part B: rows: 431-END
####################################
import pandas as pd
import seaborn as sn
import xgboost as xgb
import shap
import random
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, RocCurveDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
sn.set()

pd.set_option("display.max.columns", None)
pd.options.mode.chained_assignment = None
########################################
############ Part A ####################
########################################

### part A - section 1  Loading train data
train_df = pd.read_csv("ctr_dataset_train.csv")
print(train_df.head())

train_df = train_df.iloc[:, 1:]

# Pre Processes function definition
def pre_processes_start_df(df_to_prep):
    df_to_prep['Gender'] = df_to_prep['Gender'].astype("category")
    df_to_prep['City'] = df_to_prep['City'].astype("category")
    df_to_prep['Insurance_district'] = df_to_prep['Insurance_district'].astype("category")
    df_to_prep['Date'] = pd.to_datetime(df_to_prep['Date'], format="%d/%m/%Y")
    return df_to_prep


pre_processed_train_df = pre_processes_start_df(train_df)
pre_processed_train_df['Purchase'] = pre_processed_train_df['Purchase'].astype("category")

### part A - section 2  Data exploration
## Statistical summaries
copied_data_to_explore_df = pre_processed_train_df.copy(deep=True)
# 1: frequency of Purchase col values (yes/no) in the Data
plt.rcParams["figure.autolayout"] = True
copied_data_to_explore_df['Purchase'].value_counts().plot(kind='bar', xlabel='Purchase values', ylabel='frequency')
plt.show()

# 2: Mean Home_evaluation
print("Mean of Home_evaluation: ", copied_data_to_explore_df['Home_evaluation'].mean())
plt.hist(copied_data_to_explore_df['Home_evaluation'], 100)
plt.axvline(copied_data_to_explore_df['Home_evaluation'].mean(), color='k', linestyle='dashed', linewidth=2)
plt.show()

# 3: Top Deciles by Home_evaluation presented by City division
copied_data_to_explore_df['top_decile'] = pd.qcut(copied_data_to_explore_df['Home_evaluation'], 10, labels=False)
top_decile_df = copied_data_to_explore_df[copied_data_to_explore_df['top_decile'] == 9]
del copied_data_to_explore_df['top_decile']
plt.rcParams["figure.autolayout"] = True
top_decile_df['City'].value_counts().plot(kind='bar', xlabel='City', ylabel='frequency')
plt.show()

# 4: Mode of some categorical features: Gender, City, Insurance_district
mode_df = copied_data_to_explore_df[['Gender', 'City', 'Insurance_district']]
row = mode_df.mode().iloc[0]
print("Mode of Gender, City and Insurance_district:")
print(row)

# 5: Mean and Standard Deviation of part of data
mean_sd_data = [['Num_residents_floor', copied_data_to_explore_df['Num_residents_floor'].mean(),
                 copied_data_to_explore_df['Num_residents_floor'].std()],
                ['Floor', copied_data_to_explore_df['Floor'].mean(),
                 copied_data_to_explore_df['Floor'].std()],
                ['Home_age', copied_data_to_explore_df['Home_age'].mean(),
                 copied_data_to_explore_df['Home_age'].std()],
                ['Bedrooms_ m2', copied_data_to_explore_df['Bedrooms_ m2'].mean(),
                 copied_data_to_explore_df['Bedrooms_ m2'].std()],
                ['Living_room_m2', copied_data_to_explore_df['Living_room_m2'].mean(),
                 copied_data_to_explore_df['Living_room_m2'].std()],
                ['Garden_m2', copied_data_to_explore_df['Garden_m2'].mean(),
                 copied_data_to_explore_df['Garden_m2'].std()]]
mean_sd_data = pd.DataFrame(mean_sd_data, columns=['Feature', 'Mean', 'Standard deviation'])
print("Mean Standard Deviation Data:")
print(mean_sd_data)

#### Correlation matrix ####
# User_ID feature not included because its identifier feature
# not meaningful and irrelevant for the prediction
# Date feature also not good for us
copied_data_for_corr_mat = pre_processed_train_df.copy(deep=True)
del copied_data_for_corr_mat['User_ID']
del copied_data_for_corr_mat['Date']
corrMatrix = copied_data_for_corr_mat.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#### Should we remove features? Yes. ####
# 'User_ID' feature removed from the train dataset from same reasons above
# We will take the 'month' and 'day_of_week' values from the Date feature.
# They might have an impact on the purchase decision
#### These features will be removed after section 4


#### Target variable 'Purchase' is imbalanced ! ####
# We saw it in the first statistical summary
# Now, let's print it. count of each value (1/0) and proportion
print("value_counts of Purchase target variable:")
print(pre_processed_train_df.Purchase.value_counts())
print("Relative frequency (proportion):")
print(pre_processed_train_df.Purchase.value_counts(normalize=True))
# 0 -  count: 88101   proportion: 0.774446
# 1 -  count: 25659   proportion: 0.225554
# I chose the SMOTE over-sampling.
# SMOTE is an oversampling technique where the synthetic samples are generated for the minority class - 1.
# This algorithm helps to overcome the over-fitting problem posed by random oversampling as we'll implement next
# With the use of this technique, we should get true and better results
###### SMOTE Implementation after section 5 ######


### part A - section 3  Missing Values
## Missing values by feature
print("Missing values count in each column:")
print(pre_processed_train_df.isnull().sum(axis=0))
# I chose to do the following:
# 1. remove rows with no 'Purchase' value (2608 rows)
# I chose to do so because i think knowing the user purchase decision is basically mandatory for training.
pre_processed_train_df = pre_processed_train_df.dropna(subset=['Purchase'])

# 2. complete the rest of missing values using 3NN Imputer
###### Completion of missing values implementation will be after section 4
# I chose to impute data from known data. i think its better strategy to fill missing values
# then to delete the rows/columns.
# Each sample’s missing values are imputed using the mean value from 3 nearest neighbors found in the training set.
# Two samples are close if the features that neither is missing are close.

### part A - section 4  Feature Engineering ###
## remove unnecessary features: User_ID, Date
## Create new features from existing: month, total_m2, Home_age_rank,house_value_to_m2
## In addition, i created dummies for categorical variables

def get_home_age_rank(df):
    if df['Home_age'] < 33:
        return 'new'
    elif df['Home_age'] > 66:
        return 'old'
    else:
        return 'mid'


def feature_engineering_df(fe_df):
    del fe_df['User_ID']
    fe_df['month'] = fe_df['Date'].dt.month
    fe_df['month'] = fe_df['month'].astype('category')
    fe_df['Gender'] = fe_df['Gender'].astype('category')
    fe_df['Home_age_rank'] = fe_df.apply(get_home_age_rank, axis=1)
    fe_df['total_m2'] = fe_df['Garden_m2'] + fe_df['Living_room_m2'] + fe_df['Bedrooms_ m2']
    fe_df['Home_evaluation_for_m2'] = fe_df['Home_evaluation'] / fe_df['total_m2']
    del fe_df['Date']
    fe_df = pd.get_dummies(fe_df, columns=['Insurance_district', 'City', 'Gender', 'month', 'Home_age_rank'])
    return fe_df


print("Feature engineering...")
features_fixed_train_df = feature_engineering_df(pre_processed_train_df)


def impute_data(df_to_impute):
    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(df_to_impute)
    df_to_impute = pd.DataFrame(imputed, columns=df_to_impute.columns)
    return df_to_impute


### Completion of missing values using 3NN Imputer
print("Data imputation...")
imputeTrainData_df = impute_data(features_fixed_train_df)


def process_after_imputing(df_to_process):
    df_to_process['Floor'] = round(df_to_process['Floor'])
    df_to_process['Num_residents_floor'] = round(df_to_process['Num_residents_floor'])
    df_to_process['Home_age'] = round(df_to_process['Home_age'])
    df_to_process['Garden_m2'] = round(df_to_process['Garden_m2'])
    df_to_process['Living_room_m2'] = round(df_to_process['Living_room_m2'])
    df_to_process['Bedrooms_ m2'] = round(df_to_process['Bedrooms_ m2'])
    df_to_process['Living_room_m2'] = round(df_to_process['Living_room_m2'])
    return df_to_process


print("Data process after imputation...")
processed_imputed_train_data = process_after_imputing(imputeTrainData_df)


## SMOTE Implementation
def SMOTE_sample(df_to_sample):
    x_train_to_sample = df_to_sample.loc[:, df_to_sample.columns != 'Purchase']
    y_train_to_sample = df_to_sample['Purchase']
    over_sample = SMOTE()
    return over_sample.fit_resample(x_train_to_sample, y_train_to_sample)


print("SMOTE Implementation...")
x_train_res, y_to_train = SMOTE_sample(processed_imputed_train_data)
print(y_to_train.value_counts())
print(y_to_train.value_counts(normalize=True))


### part A - section 5  Data normalization
# I chose to normalize all data. it doesn't have Gaussian distribution
# Neural network requires
def normalize_df(df_to_norm):
    x_values = df_to_norm.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x_values)
    x_train_normalize = pd.DataFrame(x_scaled, columns=df_to_norm.columns)
    return x_train_normalize


print("Data normalization...")
x_to_train = normalize_df(x_train_res)

###############################################
############ Finish data processes ############
# Data to train is:
# x_to_train
# y_to_train
###############################################

### part A - section 6  Training
x_train, x_valid, y_train, y_valid = train_test_split(x_to_train, y_to_train, test_size=0.3, random_state=0)

############################ First model - Neural Network ############################
print("First model - Neural Network...")
NN = MLPClassifier(max_iter=100, random_state=1)
parameter_space = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],  # Activation function for the hidden layer
    'solver': ['sgd', 'adam'],  # The solver for weight optimization.
    'alpha': [0.0001, 0.05],
    # Strength of the L2 regularization term which divided by the sample size when added to the loss.
    'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule for weight updates
}

clf_NN = GridSearchCV(NN, parameter_space, n_jobs=-1, cv=3)
clf_NN.fit(x_train, y_train)

# Best parameter set
print('Best parameters found:')
print(clf_NN.best_params_)
# prediction on best params
y_pred = clf_NN.predict(x_valid)
report = classification_report(y_valid, y_pred)
print("Report: ", report)
# Report:            precision recall    f1-score support
#          0.0       0.86      0.92      0.89     26427
#          1.0       0.92      0.85      0.89     26434
#     accuracy                           0.89     52861
#    macro avg       0.89      0.89      0.89     52861
# weighted avg       0.89      0.89      0.89     52861
print("AUC: {:.4f}".format(roc_auc_score(y_valid, y_pred)))
# AUC: 0.8892


#################### Second model - XGBoost ############################
print("Second model - XGBoost...")
xgb_params = {  # Parameters for Tree Booster
    'early_stopping_rounds': 10,  # technique used to stop training when the loss on validation dataset starts increase
    'max_depth': hp.quniform("max_depth", 3, 18, 1),  # Maximum depth of a tree.
    # Increasing this value will make the model more complex and more likely to overfit.
    'gamma': hp.uniform('gamma', 1, 9),
    # Minimum loss reduction required to make a further partition on a leaf node of the tree.
    # The larger gamma is, the more conservative the algorithm will be
    'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),  # L1 regularization term on weights.
    # Increasing this value will make model more conservative.
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),  # L2 regularization term on weights.
    # Increasing this value will make model more conservative
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    # the subsample ratio of columns when constructing each tree.
    # Subsampling occurs once for every tree constructed
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    # Minimum sum of instance weight (hessian) needed in a child.
    # If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
    # then the building process will give up further partitioning
    'n_estimators': 180,  # The number of trees
    'seed': 0
}


def objective(xgb_params_obj):
    clf = xgb.XGBClassifier(
        eval_metric="logloss",
        n_estimators=xgb_params_obj['n_estimators'],
        max_depth=int(xgb_params_obj['max_depth']),
        gamma=xgb_params_obj['gamma'],
        reg_alpha=int(xgb_params_obj['reg_alpha']),
        reg_lambda=int(xgb_params_obj['reg_lambda']),
        min_child_weight=int(xgb_params_obj['min_child_weight']),
        colsample_bytree=int(xgb_params_obj['colsample_bytree']))

    eval_s = [(x_train, y_train), (x_valid, y_valid)]

    clf.fit(x_train, y_train, eval_set=eval_s, verbose=False)

    predictions = clf.predict(x_valid)
    accuracy = accuracy_score(y_valid, predictions > 0.5)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}


trials = Trials()

best_hyper_params = fmin(fn=objective,
                         space=xgb_params,
                         algo=tpe.suggest,
                         max_evals=100,
                         trials=trials)

print("The best XGBClassifier hyper parameters are:")
print(best_hyper_params)

##### XGBoost Tuned implementation #####

clf_Tuned = xgb.XGBClassifier(
    eval_metric="logloss",
    early_stopping_rounds=10,
    n_estimators=180,
    max_depth=7,
    gamma=4.373332034814524,
    reg_alpha=41,
    reg_lambda=0.8548776268291259,
    min_child_weight=1,
    colsample_bytree=0.7096002059668551)

evaluation = [(x_train, y_train), (x_valid, y_valid)]

clf_Tuned.fit(x_train, y_train, eval_set=evaluation, verbose=False)

pred_tuned = clf_Tuned.predict(x_valid)
RocCurveDisplay.from_predictions(y_valid, pred_tuned > 0.5)
plt.show()
report = classification_report(y_valid, pred_tuned > 0.5)
print("Report: ", report)
# Report:        precision    recall  f1-score   support
#          0.0       0.87      0.94      0.90     26427
#          1.0       0.93      0.85      0.89     26434
#     accuracy                           0.90     52861
#    macro avg       0.90      0.90      0.90     52861
# weighted avg       0.90      0.90      0.90     52861

print("AUC: {:.4f}".format(roc_auc_score(y_valid, pred_tuned > 0.5)))
# AUC: 0.8958

#################################################################
#################### Chosen model is XGBoost ####################
# XGBoost model gave me better performance (not too much tho...)
# That is why i chose it over NN to predict the purchase values in section 8
#################################################################

### part A - section 7  Explainable AI

# SHAP values interpret the impact of having a certain value for a given feature
# in comparison to the prediction we'd make if that feature took some baseline value.
# Fits the shap explainer
print("SHAP Explainer...")
explainer = shap.Explainer(clf_Tuned.predict, x_valid)
# Calculates the SHAP values
shap_values = explainer(x_valid)

## Global interpretability - summary_plot
# The summary plot combines feature importance with feature effects.
# Each point on the summary plot is a Shapley value for a feature and an instance.
# The position on the y-axis is determined by the feature and on the x-axis by the Shapley value.
# The color represents the value of the feature from low to high.
shap.summary_plot(shap_values)

## Local interpretability - shap_plot (im using waterfall)
shap.plots.waterfall(shap_values[random.randint(0, len(x_valid))])
shap.plots.waterfall(shap_values[random.randint(0, len(x_valid))])
shap.plots.waterfall(shap_values[random.randint(0, len(x_valid))])
shap.plots.waterfall(shap_values[random.randint(0, len(x_valid))])
shap.plots.waterfall(shap_values[random.randint(0, len(x_valid))])
# It explain why a case receives its prediction and the contributions of the predictors.
# The local interpretability enables us to pinpoint and contrast the impacts of the factors.


### part A - section 8 	Inference
print("Prediction on TEST dataset...")
test_df = pd.read_csv("ctr_dataset_test.csv")
test_df = pre_processes_start_df(test_df)
test_df = feature_engineering_df(test_df)
test_df = impute_data(test_df)
test_df = process_after_imputing(test_df)
test_df = normalize_df(test_df)
# Prediction with chosen XGBoost model
clf_XGB = xgb.XGBClassifier(
    eval_metric="logloss",
    early_stopping_rounds=10,
    n_estimators=180,
    max_depth=7,
    gamma=4.373332034814524,
    reg_alpha=41,
    reg_lambda=0.8548776268291259,
    min_child_weight=1,
    colsample_bytree=0.7096002059668551)

evaluation_set = [(x_to_train, y_to_train)]

clf_XGB.fit(x_to_train, y_to_train, eval_set=evaluation_set, verbose=False)
test_predictions = clf_XGB.predict(test_df)
# save prediction to txt file
np.savetxt(r'C:\Users\manzur\PycharmProjects\FinalProject\output_9.txt', test_predictions > 0.5, fmt='%d')
print("End Part A...")


########################################
############ Part B ####################
########################################


### part B - section 1  Loading Clustering data
clustering_df = pd.read_csv("Clustering.csv")

############### part B - sections 2&3 KMeans ###############
# First, lets use two point-view Graphs and try to estimate the right K param
# The number of "KNEES" in the plot will help us to estimate the number of clusters
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2, 10), metric='silhouette')  # (distortion metric. sum of squared)
visualizer.fit(clustering_df)
visualizer.show()
plt.show()

# It seems that 2 clusters is ok...
# let's implement it
KMmodel = KMeans(n_clusters=2).fit(clustering_df)
KM2_clustered = clustering_df.copy()
KM2_clustered.loc[:, 'Cluster'] = KMmodel.labels_

print("Let's see the division to clusters:")
KM_clust_sizes = KM2_clustered.groupby('Cluster').size().to_frame()
KM_clust_sizes.columns = ["KM_size"]
print(KM_clust_sizes)

####### part B - sections 2&3 DBSCAN #######

# First, lets check what are the two important params should be:
# eps -> defines the radius of neighborhood around a point
# MinPts -> the minimum number of neighbors within “eps” radius (2*dimensions is recommended)

# determine eps by 20 Nearest Neighbors
neighbors = NearestNeighbors(n_neighbors=20)
neighbors_fit = neighbors.fit(clustering_df)
distances, indices = neighbors_fit.kneighbors(clustering_df)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()

# eps_values to check should be 0.7 or 1.3. the latter is chosen.
# min_samples will be as the n_neighbors
DBS_clustering = DBSCAN(eps=1.3, min_samples=20).fit(clustering_df)
DBSCAN_clustered = clustering_df.copy()
DBSCAN_clustered.loc[:, 'Cluster'] = DBS_clustering.labels_

print("Let's see the division to clusters:")
DBSCAN_clust_sizes = DBSCAN_clustered.groupby('Cluster').size().to_frame()
DBSCAN_clust_sizes.columns = ["DBSCAN_size"]
print(DBSCAN_clust_sizes)


####### part B - section 4 #######

# First metrics: Silhouette Score
# Lets display a measure of how close each point in a cluster is to points in the neighbouring clusters.
# The higher the Silhouette Coefficients (the closer to +1),
# the further away the cluster’s samples are from the neighbouring clusters samples.

# K-MEANS
print("K-MEANS Silhouette Coefficient: %0.3f" % silhouette_score(clustering_df, KMmodel.labels_))
# Silhouette Coefficient: 0.104 ----> not so good,but better than k=3,4...

# DBSCAN
print("DBSCAN Silhouette Coefficient: %0.3f" % silhouette_score(clustering_df, DBS_clustering.labels_))
# Silhouette Coefficient: -0.018 ----> not good
# It indicate that negative samples might have been assigned to the wrong cluster

### Second metric: Calinski-Harabasz Index ###
# The score is defined as the ratio between the within-cluster dispersion and the between-cluster dispersion.
# The higher the Index, the better the performance.

# K-MEANS
print("K-MEANS Calinski-Harabasz score: %0.3f" % calinski_harabasz_score(clustering_df, KMmodel.labels_))
# Score: 295.090

# DBSCAN
print("DBSCAN Calinski-Harabasz score: %0.3f" % calinski_harabasz_score(clustering_df, DBS_clustering.labels_))
# Score: 106.712

####################
# Conclusions:
# it looks like DBSCAN failed to generate reasonable clusters
# on the other end, K-MEANS leads to 2 clusters, with better Silhouette & Calinski-Harabasz Scores
# it is better then DBSCAN but not good as i expected
####################

############ part B - section 5 ###############
# Clusters plots

# KMeans clusters plot
pca_num_components = 2
reduced_data = PCA(n_components=pca_num_components).fit_transform(KM2_clustered)
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])
sn.scatterplot(x="pca1", y="pca2", hue=KM2_clustered['Cluster'], data=results)
plt.title('K-means Clustering')
plt.show()

# DBSCAN clusters plot
pca_num_components = 2
reduced_data = PCA(n_components=pca_num_components).fit_transform(DBSCAN_clustered)
results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])
sn.scatterplot(x="pca1", y="pca2", hue=DBSCAN_clustered['Cluster'], data=results)
plt.title('DBSCAN Clustering with 2 dimensions')
plt.show()