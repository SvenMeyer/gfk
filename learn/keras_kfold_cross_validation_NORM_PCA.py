'''


Problem : INFO:tensorflow:Summary name dense_1_W:0 is illegal; using dense_1_W_0 instead.
'''


import os
import glob
import time
import pandas as pd
import numpy
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing, decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

print("keras.__version__ = ", keras.__version__)
print("tensorflow.__version__ = ", tf.__version__)

# import matplotlib.pyplot as plt

TEST = False
PCA  = True
PCA_comp = 256

HOME_DIR = "/media/sf_SHARE"
if not os.path.isdir(HOME_DIR):
    HOME_DIR = os.path.expanduser("~")
LOT_DIR  = "/ML_DATA/GFK/DE/Lotame/"
LOT_FILE = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738"
GXL_DIR  = "/ML_DATA/GFK/DE/Hyperlane/unimputed-target-groups/2017-02-01/"
GXL_FILE = "GXL_data.tsv"
TARGET   = "dep_tg_bin_gender_1"
# TARGET   = "dep_tg_bin_pet_owner_209"

if TEST:
    LOT_FILE = "Lot_data_test"
    GXL_FILE = "GXL_data_test.tsv"


COL_NAME_PID ="pnr"

df_GXL = pd.read_csv(HOME_DIR + GXL_DIR + GXL_FILE, sep='\t', dtype=numpy.int32)
print("df_GXL.shape = ", df_GXL.shape)
# print(df_GXL.iloc[:10,:5])

#df_Lot = pd.read_pickle(HOME_DIR + LOT_DIR + LOT_FILE + ".pkl")
df_Lot = pd.read_csv(HOME_DIR + LOT_DIR + LOT_FILE + ".tsv", sep='\t') #, dtype=np.int32)
print("df_Lot.shape = ", df_Lot.shape)
# print(df_Lot.iloc[:10,:5])
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

df = pd.merge(df_Lot, df_GXL[[COL_NAME_PID, TARGET]], on=COL_NAME_PID, how='inner')
print(df.shape)
# print(df)

# df = df[df[TARGET] != -1]
df.drop(df.index[df[TARGET] == -1], inplace=True) # df[TARGET] = df.iloc[:,-1]
print("after dropping NO DATA rows : df.shape =", df.shape)

X_data = df.iloc[:,2:-1].values.astype('float32')

if NORM:
    norm_scale = preprocessing.Normalizer(copy=False).fit(X_data)
    X_data = norm_scale.transform(X_data)
else:
    std_scale = preprocessing.StandardScaler(copy=False).fit(X_data)
    X_data = std_scale.transform(X_data)
    if PCA:
        print("Start PCA", PCA_comp," ... ", end="")
        pca = decomposition.PCA(n_components=PCA_comp)
        pca.fit(X_data)
        X_data = pca.transform(X_data)
        print("DONE in ", (time.time() - start), "sec")
    
print("X_data.shape = ", X_data.shape)

Y_data = df.iloc[:,-1].values.astype('float32').reshape(-1, 1)  # reshape because StandardScaler does not accept 1d arrays any more
minmax_scale = preprocessing.MinMaxScaler().fit(Y_data)
Y_data = minmax_scale.transform(Y_data).ravel()                 # ravel back to 1d array

X, X_test, Y, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=1)

seed = 7
numpy.random.seed(seed)

n_layer_1 = X.shape[1]
n_layer_2 = 2**math.ceil(math.log2(math.sqrt(n_layer_1)))
print("n_layer_1 = ", n_layer_1)
print("n_layer_2 = ", n_layer_2)

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(X.shape[1], input_dim=X.shape[1], kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(64, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
'''
# https://sites.google.com/site/xindongsite/home/kerasrunning5callbacks
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
'''
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
cvscores = []
# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10, verbose=2)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
start = time.time()

estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, nb_epoch=300, batch_size=16, verbose=1))) # , callbacks=calls
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, fit_params={'mlp__callbacks':callbacks_list})

# http://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
# model.fit(X_train.astype('float32'), Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#      shuffle=True, verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)

# scores = cross_val_score(model, X, Y, cv=kfold, scoring=None) # fit_params callbacks=callbacks_list

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("DONE in ", (time.time() - start), "sec")

print('scores:')
print(scores)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % ("???", scores[1]*100))
cvscores.append(scores[1] * 100)
# print(history.history.keys())  # summarize history for accuracy
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

print("evaluate TEST-set (out of training set)")
print("confusion_matrix")
Y_predict = model.predict(X_test)>=0.5
print(confusion_matrix(Y_predict,Y_test))
print("cohen_kappa_score = ", cohen_kappa_score(Y_test, Y_predict))
print("accuracy_score    = ",    accuracy_score(Y_test, Y_predict))

# serialize model to JSON
model_json = model.to_json()
with open(HOME_DIR+'/ML_DATA/GFK/model/lotame_model.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(HOME_DIR+'/ML_DATA/GFK/model/lotame_model_weights.h5')
print("Saved model to disk")

'''
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''