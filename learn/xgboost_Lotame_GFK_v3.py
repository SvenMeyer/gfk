# v3
# drop duplicate columns in df_Lot (before removing rows)
# join X_data with all Y_data target 
# loop over all Y-data targets
'''
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
'''
import os
import glob
import time
import pandas as pd
import numpy
import math

from xgboost import XGBClassifier

# import tensorflow as tf
# print("tensorflow.__version__ = ", tf.__version__)

# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# print("keras.__version__ = ", keras.__version__)

from sklearn import preprocessing, decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

from matplotlib import pyplot as plt

import pickle

TEST = True
NORM = True
SCALE= False
PCA  = False
PCA_comp = 256

HOME_DIR = "/media/sf_SHARE"
if not os.path.isdir(HOME_DIR):
    HOME_DIR = os.path.expanduser("~")
LOT_DIR = "/ML_DATA/GFK/DE/Lotame/"
LOT_FILE = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738"
LOT_FILE_no_index_cols = 2
GXL_DIR = "/ML_DATA/GFK/DE/Hyperlane/unimputed-target-groups/2017-02-01/"
GXL_FILE = "data.tsv"
TARGET   = "dep_tg_bin_gender_1"
# TARGET   = "dep_tg_bin_pet_owner_209"
# TARGET   = "dep_tg_bin_buyer_baby_products_227"

if TEST:
    LOT_FILE = "Lot_data_test"
    GXL_FILE = "GXL_data_test.tsv"

COL_NAME_PID = "pnr"

# read GfK panel data
df_GXL = pd.read_csv(HOME_DIR + GXL_DIR + GXL_FILE, sep='\t', dtype=numpy.int32)
print("df_GXL.shape = ", df_GXL.shape)
# print(df_GXL.iloc[:10,:5])

# read Lotame Internet behavior training data
# df_Lot = pd.read_pickle(HOME_DIR + LOT_DIR + LOT_FILE + ".pkl")
df_Lot = pd.read_csv(HOME_DIR + LOT_DIR + LOT_FILE + ".tsv", sep='\t')  # , dtype=np.int32)
print("df_Lot.shape = ", df_Lot.shape)
Lot_no_features = df_Lot.shape[1] - LOT_FILE_no_index_cols
print("Lot_no_features =", Lot_no_features)

# drop duplicate columns
df_Lot = df_Lot.T.drop_duplicates().T
print("dropped duplicate columns : df_Lot.shape =", df_Lot.shape)

df_column_names=pd.DataFrame(df_Lot.columns)
df_column_names.to_csv(HOME_DIR+"/ML_DATA/GFK/DE/model/column_names.tsv", sep='\t')

'''
# Pre-Processing
# print("events in Lot table = ", df_Lot.sum(numeric_only=True).sum()) # need to exclude pnr
print("histogram : no of LOTBEH with 0 .. 99 panel visitor")
el = df_Lot.astype(bool).sum(axis=0)      # count no of non-zero entries for each LOTBEH
el[el<100].hist(bins=100, figsize=(10,10))
# drop LOTBET columns with very few entries (panel members)
MIN_panel_per_LOTBEH = 4
df_Lot.drop(el.index[el < MIN_panel_per_LOTBEH], axis=1, inplace=True)
print("removed LOTBEH columns with less than",  MIN_panel_per_LOTBEH, "entries (panel members) - df_Lot.shape = ", df_Lot.shape)

ep = df_Lot.astype(bool).sum(axis=1)
ep[ep<100].hist(bins=100, figsize=(10,10))
'''
# merge X and Y DataFrames ###########################################################################################################

# df_data = pd.merge(df_Lot, df_GXL[[COL_NAME_PID, TARGET]], on=COL_NAME_PID, how='inner') # merge only TARGET column
df_data = pd.merge(df_Lot, df_GXL, on=COL_NAME_PID, how='inner')
print(df_data.shape)
print(df_data)
exit()
sxc = LOT_FILE_no_index_cols # start X-data column
exc = LOT_FILE_no_index_cols + Lot_no_features # end X-data column

# only keep rows (= users) with min_events
if TEST:
    min_events = 1
else:
    min_events = 100
low_events = df.iloc[:,LOT_FILE_no_index_cols:].sum(axis=1) < min_events
df_low_events = df[low_events]
df = df[~low_events]
print("dropped rows with with less than", min_events, "events : df.shape =", df.shape)

# for TARGET in df_data.columns[exc:] # loop over all TARGET columns ################################################################
print("TARGET = ", TARGET)
df = df_data.drop(df_data.index[df_data[TARGET] == -1])
print("only use rows where TARGET != -1 : df.shape =", df.shape)



# def pre_process_X(df): # return numpy_array





X_data = df.iloc[:, LOT_FILE_no_index_cols:-1].values.astype('float32')

if NORM:
    norm_scale = preprocessing.Normalizer(copy=False).fit(X_data)
    X_data = norm_scale.transform(X_data)
else:
    if SCALE:
        std_scale = preprocessing.StandardScaler(copy=False).fit(X_data)
        X_data = std_scale.transform(X_data)

if PCA:
    print("Start PCA", PCA_comp, " ... ", end="")
    start = time.time()
    pca = decomposition.PCA(n_components=PCA_comp)
    pca.fit(X_data)
    X_data = pca.transform(X_data)
    print("DONE in ", (time.time() - start), "sec")

print("X_data.shape = ", X_data.shape)

Y_data = df.iloc[:, -1].values.astype('float32').reshape(-1, 1)  # reshape because StandardScaler does not accept 1-D arrays any more
minmax_scale = preprocessing.MinMaxScaler().fit(Y_data)
Y_data = minmax_scale.transform(Y_data).ravel()  # ravel back to 1-D array

# t-sne visualization
'''
def plot_embedding(x, y):
    cm = plt.cm.get_cmap('RdYlGn')
    f = plt.figure(figsize=(13, 13))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=y, cmap=cm)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.show()

import sklearn.manifold
# weights = model.get_layer(index={your layer index}).get_weights()
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0, verbose=1)
X_data_tsne = tsne.fit_transform(X_data)
# plot_embedding(X_data_tsne, Y_data)  # >> crashed following XGBoost !!
'''

X, X_test, Y, Y_test = train_test_split(X_data, Y_data, test_size=0.25, random_state=1)

label_encoded_y = Y

if TEST:
    exit()

model = XGBClassifier()

n_estimators  = [200] # [100, 200]
max_depth     = [2] # [2, 3, 4]
learning_rate = [0.1] # [0.3, 0.2, 0.1, 0.03, 0.01]
colsample_bytree = [1.0]
subsample =        [1.0]
min_child_weight = [0.95] # [0.9, 0.95]
scale_pos_weight = [1.00]
reg_lambda       = [0.90] # [0.85, 0.90, 0.95, 1.00]
reg_alpha        = [0.00] # [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

param_grid = dict(max_depth       = max_depth,
                  n_estimators    = n_estimators,
                  learning_rate   = learning_rate,
                  colsample_bytree= colsample_bytree,
                  subsample       = subsample,
                  min_child_weight= min_child_weight,
                  scale_pos_weight= scale_pos_weight,
                  reg_lambda = reg_lambda,
                  reg_alpha = reg_alpha)
# objective="binary:logistic"
'''
fit_params={"early_stopping_rounds":42, 
            "eval_metric" : "mae", 
            "eval_set" : [[testX, testY]]}
'''

kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=2)

print("*** starting grid_search.fit ***")
grid_result = grid_search.fit(X, label_encoded_y)

model = grid_result.best_estimator_
print("best model = ", model)

cv_results = pd.DataFrame.from_dict(grid_result.cv_results_)
cv_results.sort_values('rank_test_score', inplace=True)
# print("cv_results:")
# print(cv_results)
cv_results.to_csv(HOME_DIR+"/ML_DATA/GFK/DE/model/xgb_model_cv_results_" + TARGET + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + ".tsv", sep='\t')

# save best model
file_out = HOME_DIR+"/ML_DATA/GFK/DE/model/xgb_model_"+TARGET+".pkl"
print("save model to : ", file_out, end='')
pickle.dump(model, open(file_out, "wb"))
print(" ... DONE")

print("*** results ***")

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("*** evaluate training/validation-set")
y_pred_train = model.predict(X)
print("confusion_matrix")
print(                         confusion_matrix(y_pred_train, label_encoded_y))
print("cohen_kappa_score = ", cohen_kappa_score(y_pred_train, label_encoded_y))
print("accuracy_score    = ",    accuracy_score(y_pred_train, label_encoded_y))

print("*** evaluate TEST-set (out of training set)")
y_pred_test = model.predict(X_test)
print("confusion_matrix")
print(                         confusion_matrix(y_pred_test, Y_test))
print("cohen_kappa_score = ", cohen_kappa_score(y_pred_test, Y_test))
print("accuracy_score    = ",    accuracy_score(y_pred_test, Y_test))

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
axes[0].set_title('TARGET = 0')
axes[1].set_title('TARGET = 1')
pd.DataFrame(model.predict(X_test[Y_test==0])).hist(bins=20, ax=axes[0])
pd.DataFrame(model.predict(X_test[Y_test==1])).hist(bins=20, ax=axes[1])

'''
# plot_importance(model) # not available within sklearn wrapper of XGBoost
# pyplot.show()
print("*** draw feature importance bar **")
fi = model.feature_importances_
print("model.feature_importances_ = ", fi)
pyplot.bar(range(len(fi)), fi)
pyplot.show()


fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
# Axes3D.plot_surface(X, Y, Z, *args, **kwargs)
# Axes3D.plot_surface(n_estimators, max_depth, Z, *args, **kwargs)

# plot results trees x depth
scores = [x[1] for x in grid_result.grid_scores_]
scores = numpy.array(scores).reshape(len(max_depth), len(n_estimators), len(learning_rate))
for i, value in enumerate(max_depth):
    pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
pyplot.legend()
pyplot.xlabel('n_estimators')
pyplot.ylabel('Log Loss')
pyplot.savefig('n_estimators_vs_max_depth.png')

# plot results trees x learning rate
scores = [x[1] for x in grid_result.grid_scores_]
scores = numpy.array(scores).reshape(len(learning_rate), len(n_estimators), len(learning_rate))
for i, value in enumerate(learning_rate):
    pyplot.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
pyplot.legend()
pyplot.xlabel('n_estimators')
pyplot.ylabel('Log Loss')
pyplot.savefig('n_estimators_vs_learning_rate.png')
'''
