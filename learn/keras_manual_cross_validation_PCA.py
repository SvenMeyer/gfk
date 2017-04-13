'''


Problem : INFO:tensorflow:Summary name dense_1_W:0 is illegal; using dense_1_W_0 instead.
'''


import os
import glob
import time
import pandas as pd
import numpy
import math
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.noise import GaussianNoise
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

print("keras.__version__ = ", keras.__version__)
print("tensorflow.__version__ = ", tf.__version__)

# import matplotlib.pyplot as plt

TEST = False
NORM = True
PCA  = False
PCA_comp = 4096

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

if NORM: # scale sum(row) = 1
    norm_scale = preprocessing.Normalizer(copy=False).fit(X_data)
    X_data = norm_scale.transform(X_data)
else:    # scale sum(col) = 1
    std_scale = preprocessing.StandardScaler(copy=False).fit(X_data)
    X_data = std_scale.transform(X_data)

if PCA:
    print("Start PCA", PCA_comp," ... ", end="")
    start = time.time()
    pca = decomposition.PCA(n_components=PCA_comp)
    pca.fit(X_data)
    X_data = pca.transform(X_data)
    print("DONE in ", (time.time() - start), "sec")
    
print("X_data.shape = ", X_data.shape)

# scale Y (target) to 0..1
Y_data = df.iloc[:,-1].values.astype('float32').reshape(-1, 1)  # reshape because StandardScaler does not accept 1d arrays any more
minmax_scale = preprocessing.MinMaxScaler().fit(Y_data)
Y_data = minmax_scale.transform(Y_data).ravel()                 # ravel back to 1d array

X, X_test, Y, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=1)

seed = 7
numpy.random.seed(seed)

# Tensorboard callback
tbCallback = keras.callbacks.TensorBoard(log_dir=HOME_DIR+'/ML_DATA/Tensorboard', histogram_freq=10, write_graph=True, write_images=True)
# checkpoint
# filepath=HOME_DIR+"/ML_DATA/GFK/model/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
# checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint, tbCallback]
callbacks_list = [tbCallback]

n_layer_1 = X.shape[1]
n_layer_2 = 2**math.ceil(math.log2(math.sqrt(n_layer_1)))
print("n_layer_1 = ", n_layer_1)
print("n_layer_2 = ", n_layer_2)

noise_stddev = 1.0
for n in range(26):
    i = 1
    print("noise_stddev =",noise_stddev)
    no_splits=4
    with tf.device("/gpu:0"):
        kfold = StratifiedKFold(n_splits=no_splits, shuffle=True) #, random_state=seed)
        cvscores = []
        for train, validate in kfold.split(X, Y):
            print(i ,'/' , no_splits , "--------------------")
            start = time.time()
          # create model
            model = Sequential()
            model.add(Dense(n_layer_1, input_dim=n_layer_1, kernel_initializer='glorot_uniform', activation='relu'))
            model.add(Dense(n_layer_2, kernel_initializer='glorot_uniform', activation='relu'))
            model.add(GaussianNoise(stddev=noise_stddev))
            model.add(Dense(        1, kernel_initializer='glorot_uniform', activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
            # Fit the model
            history = model.fit(X[train], Y[train], epochs=200, batch_size=10, verbose=0, callbacks=callbacks_list)
            print("DONE in ", (time.time() - start), "sec")
            # evaluate the model
            scores = model.evaluate(X[validate], Y[validate], verbose=2)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            print(history.history.keys())  # summarize history for accuracy
            Y_predict = model.predict(X[validate]) >= 0.5
            print("cohen_kappa_score = ", cohen_kappa_score(Y_predict,Y[validate]))
            print("accuracy_score    = ",    accuracy_score(Y_predict,Y[validate]))
            i += 1
            noise_stddev = noise_stddev / math.sqrt(2)

            print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
            
            print("evaluate TEST-set (out of training set)")
            print("confusion_matrix")
            print(confusion_matrix(model.predict(X_test)>=0.5,Y_test))
            print("cohen_kappa_score = ", cohen_kappa_score(model.predict(X_test)>=0.5,Y_test))
            print("accuracy_score    = ", accuracy_score(model.predict(X_test)>=0.5,Y_test))

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