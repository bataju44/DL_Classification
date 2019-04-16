
# coding: utf-8

# In[ ]:




# In[45]:

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().magic('matplotlib inline')

sig_file = "SIG.csv"
bkg_file = "BKG.csv"


# In[46]:

variable = ["lepton_pt","pt_leading_jets","pt_second_jets","num_jet","Met1","Met0","MT","dR_lep_jet","eta_lepton","phi_lepton","eta_leading_jet","phi_leading_jet","eta_subleading_jet","phi_subleading_jet","phi_met"]
v= ["lepton_pt","pt_leading_jets","pt_second_jets","num_jet","Met1","Met0","MT","dR_lep_jet","eta_lepton","phi_lepton","eta_leading_jet","phi_leading_jet","eta_subleading_jet","phi_subleading_jet","phi_met","signal"]


# In[47]:

df_sig = pd.read_csv(sig_file,index_col=0)
df_bkg = pd.read_csv(bkg_file,index_col=0)


# In[48]:

#tagging signal
df_sig["signal"] = 1
df_sig


# In[49]:

df_bkg["signal"]= 0
df_bkg


# In[ ]:




# In[50]:

for var in df_sig.columns:
    print var
    plt.figure()
    plt.hist(df_sig[var],bins="auto",histtype="step", color="red",label="signal",normed=True,stacked=True)
    plt.hist(df_bkg[var],bins="auto",histtype="step", color="orange",label="bkg",normed=True,stacked=True)
    plt.legend(loc='upper right')
    plt.show()


# ##Normalization 

# In[51]:

min_max_scaler = preprocessing.MinMaxScaler()

def normalize_stuff(n):
    return (n - n.min())/(n.max() - n.min())


# In[52]:

series_list = []
 
for var in df_sig.columns:
    print var
    if var == "signal":
        series_list.append(df_sig[var])
        continue
    #print df_sig[var] 
    print df_sig[var].min()
    print df_sig[var].max()
    #print normalize_stuff(df_sig[var])
    print series_list.append((df_sig[var] - df_sig[var].min())/(df_sig[var].max() - df_sig[var].min()))




df_norm_sig = pd.DataFrame(series_list).T


# In[53]:

series_list_bkg = []
for var in df_bkg.columns:
    print var
    if var == "signal":
        series_list_bkg.append(df_bkg[var])
        continue
    #print df_sig[var] 
    print df_bkg[var].min()
    print df_bkg[var].max()
    #print normalize_stuff(df_sig[var])
    print series_list_bkg.append((df_bkg[var] - df_bkg[var].min())/(df_bkg[var].max() - df_bkg[var].min()))

df_norm_bkg = pd.DataFrame(series_list_bkg).T


# In[54]:

df_norm_bkg


# In[55]:

for var in df_sig.columns:
    print var
    plt.figure()
    plt.hist(df_norm_sig[var],bins="auto",histtype="step", color="red",label="signal",normed=True,stacked=True)
    plt.hist(df_norm_bkg[var],bins="auto",histtype="step", color="orange",label="bkg",normed=True,stacked=True)
    plt.legend(loc='upper right')
    plt.show()


# All parameters are used

# In[56]:

#using 80% of the data for training 
n_bkg = 94679*80/100
n_sig = 95623*80/100

df_train = pd.concat([df_norm_sig.iloc[:n_sig],df_norm_bkg.iloc[:n_bkg]])
df_test =  pd.concat([df_norm_sig.iloc[n_sig:],df_norm_bkg.iloc[n_bkg:]])

df_train = sklearn.utils.shuffle(df_train)
df_test = sklearn.utils.shuffle(df_test)



x_train = df_train[variable].reset_index(drop=True)
y_train = df_train["signal"].reset_index(drop=True)


x_test = df_test[variable].reset_index(drop=True)
y_test = df_test["signal"].reset_index(drop=True)


# In[ ]:




# In[57]:

print np.shape(x_train)
print np.shape(y_test)


# 

# In[ ]:




# In[58]:

x_train.shape[1]


# In[59]:

import os 
print "Using Queue:", os.environ["PBS_QUEUE"]
gpuid=int(os.environ["PBS_QUEUE"][3:4])
print "Using GPU:", gpuid
os.environ['THEANO_FLAGS'] = "device=cuda,floatX=float32,force_device=True" 



# In[17]:

import theano


# In[62]:

from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout


model = Sequential()
model.add(Dense(15, input_dim=x_train.shape[1], kernel_initializer="random_uniform", activation='tanh'))
model.add(Dense(32, kernel_initializer="random_uniform", activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer="random_uniform", activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_initializer="random_uniform", activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,kernel_initializer="random_uniform", activation='sigmoid'))


# In[63]:

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:




# In[64]:

history = model.fit(x_train.values,y_train.values,validation_data=(x_test.values,y_test.values), epochs=1000, batch_size=128)


# In[65]:

print history.history


# In[66]:

model.metrics_names
model.evaluate(x_test.values,y_test.values,batch_size=32)


# In[ ]:




# In[67]:

y_predit = model.predict(x_test.values, batch_size=32)


# signal to background prediction 

# In[68]:

#this is the number of signal events in the training and test  dataset
print y_train.sum()
print y_test.sum()

plt.figure()
plt.hist(y_test)
plt.show()


# In[69]:

print y_predit


plt.figure()
plt.hist(y_predit)
plt.show()


# In[70]:

for u in history.history:
    plt.figure()
    plt.plot(history.history[u])
    plt.title(u)


# In[71]:


from sklearn.metrics import auc
from sklearn.metrics import roc_curve


y_predit = model.predict(x_test.values).ravel()

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_predit)


auc_model = auc(fpr_keras, tpr_keras)


# In[ ]:




# In[72]:

plt.figure()

plt.plot(fpr_keras, tpr_keras, label='Tree (area = {:.3f})'.format(auc_model))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[73]:

test_file = '1600sig.csv'
test_data = pd.read_csv(test_file,index_col=0)
test_array = test_data[variable].reset_index(drop=True)


# In[74]:

series_list_test = []
 
for var in test_data.columns:
    print var
    if var == "signal":
        series_list_test.append(test_data[var])
        continue
    #print df_sig[var] 
    print test_data[var].min()
    print test_data[var].max()
    #print normalize_stuff(df_sig[var])
    print series_list_test.append((test_data[var] - test_data[var].min())/(test_data[var].max() - test_data[var].min()))




df_norm_sig_test = pd.DataFrame(series_list_test).T
print df_norm_sig_test
test_signal = np.ones(4954)
print test_signal


# In[75]:

test_predict = model.predict(df_norm_sig_test.values).ravel()
fpr_, tpr_, thresholds_ = roc_curve(test_signal, test_predict)
auc_test = auc(fpr_, tpr_)


plt.figure()
plt.hist(test_predict)
plt.show()


# In[49]:

plt.figure()

plt.plot(fpr_, tpr_, label='Area under the curve (area = {:.3f})'.format(auc_test))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# #Decision Tree

# In[31]:

from sklearn import tree

n_bkg = 94679*80/100
n_sig = 95623*80/100

df_train = pd.concat([df_sig.iloc[:n_sig],df_bkg.iloc[:n_bkg]])
df_test =  pd.concat([df_sig.iloc[n_sig:],df_bkg.iloc[n_bkg:]])

df_train = sklearn.utils.shuffle(df_train)
df_test = sklearn.utils.shuffle(df_test)



x_train = df_train[variable].reset_index(drop=True)
y_train = df_train["signal"].reset_index(drop=True)


x_test = df_test[variable].reset_index(drop=True)
y_test = df_test["signal"].reset_index(drop=True)


# In[47]:

tree_model = tree.DecisionTreeClassifier()
tree_model = tree_model.fit(x_train,y_train)


# In[43]:


from sklearn.metrics import auc
from sklearn.metrics import roc_curve
y_predict_tree = tree_model.predict(x_test)
fpr,tpr,t = roc_curve(y_test,y_predict_tree)
auc_tree = auc(fpr,tpr)



# In[45]:

plt.figure()

plt.plot(fpr, tpr, label='Tree (area = {:.3f})'.format(auc_tree))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[50]:

#from https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting
from sklearn.ensemble import GradientBoostingClassifier

GDC = GradientBoostingClassifier(n_estimators=500, learning_rate=0.6,
                                 max_depth=5, random_state=0).fit(x_train, y_train)
GDC.score(x_test,y_test)


# In[51]:

y_predict_gdc = GDC.predict(x_test)
fpr_gdc,tpr_gdc,t = roc_curve(y_test,y_predict_gdc)
auc_gdc = auc(fpr_gdc,tpr_gdc)

plt.figure()

plt.plot(fpr_gdc, tpr_gdc, label=' GDC Tree (area = {:.3f})'.format(auc_gdc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[42]:

from sklearn.ensemble import ExtraTreesClassifier 

tree_2 = ExtraTreesClassifier(verbose=1,n_estimators=50)


# In[43]:

tree_2.fit(x_train,y_train)


# In[44]:

tree_2.score(x_test,y_test)


# trying with a shallow network

# In[18]:

from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout


model1 = Sequential()
model1.add(Dense(15, input_dim=x_train.shape[1], kernel_initializer="random_uniform", activation='tanh'))
model1.add(Dense(32, kernel_initializer="random_uniform", activation='relu'))
model1.add(Dense(1,kernel_initializer="random_uniform", activation='sigmoid'))


# In[19]:

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()


# In[20]:

history1 = model1.fit(x_train.values,y_train.values,validation_data=(x_test.values,y_test.values), epochs=50, batch_size=32)


# In[21]:

model1.metrics_names
model1.evaluate(x_test.values,y_test.values,batch_size=32)


# In[22]:

y_predit1 = model1.predict(x_test.values, batch_size=32)


# In[23]:

print y_predit1


plt.figure()
plt.hist(y_predit1)
plt.show()


# In[25]:

for u in history1.history:
    plt.figure()
    plt.plot(history1.history[u])
    plt.title(u)


# In[28]:


from sklearn.metrics import auc
from sklearn.metrics import roc_curve


fpr_1, tpr_1, thresholds_keras = roc_curve(y_test, y_predit1)


auc_model1 = auc(fpr_1, tpr_1)


# In[29]:

plt.figure()

plt.plot(fpr_1, tpr_1, label='Tree (area = {:.3f})'.format(auc_model1))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[ ]:



