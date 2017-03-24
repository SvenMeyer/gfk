import os
import glob
import time
import pandas as pd
import numpy
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from sklearn import decomposition
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

# import matplotlib.pyplot as plt

TEST = False

HOME_DIR = "/media/sf_SHARE"
if not os.path.isdir(HOME_DIR):
    HOME_DIR = os.path.expanduser("~")
LOT_DIR  = "/ML_DATA/GFK/DE/Lotame/"
LOT_FILE = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738"
GXL_DIR  = "/ML_DATA/GFK/DE/Hyperlane/unimputed-target-groups/2017-02-01/"
GXL_FILE = "GXL_data.tsv"
TARGET   = "dep_tg_bin_gender_1"
TARGET   = "dep_tg_bin_pet_owner_209"

if TEST:
    LOT_FILE = "Lot_data_test"
    GXL_FILE = "GXL_data_test.tsv"

COL_NAME_PID ="pnr"

df_GXL = pd.read_csv(HOME_DIR + GXL_DIR + GXL_FILE, sep='\t', dtype=numpy.int32)
print("df_GXL.shape = ", df_GXL.shape)
print(df_GXL.iloc[:10,:5])

#df_Lot = pd.read_pickle(HOME_DIR + LOT_DIR + LOT_FILE + ".pkl")
df_Lot = pd.read_csv(HOME_DIR + LOT_DIR + LOT_FILE + ".tsv", sep='\t') #, dtype=np.int32)
print("df_Lot.shape = ", df_Lot.shape)
print(df_Lot.iloc[:10,:5])
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
df.drop(df.index[df[TARGET] == -1], inplace=True)
print("after dropping NO DATA rows : df.shape =", df.shape)

# extract features and target into numpy ndarry
X_data = df.iloc[:,2:-1].values.astype('float32')
Y_data = df.iloc[:,-1].values.astype('float32')

pca = decomposition.PCA(n_components=128)
pca.fit(X_data)
X_data = pca.transform(X_data)
print("X_data.shape = ", X_data.shape)

Y_data = Y_data - 1

X, X_test, Y, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=1)

seed = 7
numpy.random.seed(seed)

start = time.time()
with tf.device('/cpu:0'):
    kfold = StratifiedKFold(n_splits=10, shuffle=True) #, random_state=seed)
    cvscores = []
    for train, validate in kfold.split(X, Y):
      # create model
        model = Sequential()
        model.add(Dense(X.shape[1], input_dim=X.shape[1], init='glorot_uniform', activation='relu'))
        model.add(Dense(32, init='glorot_uniform', activation='relu'))
        model.add(Dense( 1, init='glorot_uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        history = model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=1)
        print("DONE in ", (time.time() - start), "sec")
        # evaluate the model
        scores = model.evaluate(X[validate], Y[validate], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        print(history.history.keys())  # summarize history for accuracy
        print("cohen_kappa_score = ", cohen_kappa_score(model.predict(X[validate])>=0.5,Y[validate]))
        print("accuracy_score    = ", accuracy_score(model.predict(X[validate])>=0.5,Y[validate]))
        
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

print("evaluate TEST-set (out of training set)")
print("confusion_matrix")
print(confusion_matrix(model.predict(X_test)>=0.5,Y_test))
print("cohen_kappa_score = ", cohen_kappa_score(model.predict(X_test)>=0.5,Y_test))
print("accuracy_score    = ", accuracy_score(model.predict(X_test)>=0.5,Y_test))

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