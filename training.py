from osgeo import ogr, gdal, osr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sbn

#gdal.AllRegister()


import pickle
# %% 
    
pickle_in = open(r"butun.pickle","rb")
concatenated = pickle.load(pickle_in)

# pickle_in = open("yanmamis.pickle","rb")
# df2 = pickle.load(pickle_in)



# concatenated = pd.concat([df1, df2], axis=0)


# pickle_out = open("butun.pickle","wb")
# pickle.dump(concatenated, pickle_out)
# pickle_out.close()


x= concatenated.iloc[:,:12]
y= concatenated.iloc[:,-1]

y=y.to_numpy()



# y=y.reshape(x.shape[0],1)
print(y.shape)

# %% normalization
#X = (x - np.min(x))/(np.max(x)-np.min(x)).values
X=x


# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

#scikitlearn kütüphanesi bölümünde kullanılacakları _sk ile isimlendirdim
x_train_sk = x_train
x_test_sk = x_test
y_train_sk = y_train
y_test_sk = y_test

#scikitlearn kütüphanesi harici kullanabilmek için Transpozlarını aldım
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

#good

# %% parameter initialize and sigmoid function
# dimension = 30
def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b


# w,b = initialize_weights_and_bias(30)

def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head
# print(sigmoid(0))

# %%
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients

#%% Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

#%%  # prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

# %% logistic_regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
#logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 500)    





"""
#%% sklearn with LR
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train_sk,y_train_sk.T)
print("scikit learn test accuracy {}".format(model.score(x_test_sk,y_test_sk.T)))


pkl_filename = "model.pickle"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)


agirliklar=model.coef_


i=1
for a in agirliklar:
    for b in a:
        print("band ",i," ",b)
        i+=1

"""

#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

MinMaxScaler(copy=True, feature_range=(0,1))

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model



model = Sequential()

model.add(Dense(12,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12,activation='relu'))
model.add(Dropout(0.2))





model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', 
              loss=tf.keras.losses.BinaryCrossentropy(), 
              metrics=['accuracy'])






log_dir = "logs/fit/"

# my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=50),
#               #tf.keras.callbacks.ModelCheckpoint('model.{epoch:02d}--{val_loss:.2f}.h5'),
#               tf.keras.callbacks.ModelCheckpoint(filepath='eniyi_model.keras',verbose=1,save_best_only=True),
#               tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)             

#               ]



my_callbacks = [
    # EarlyStopping: Eğitim sürecini val_loss izleyerek 50 epoch boyunca gelişme olmazsa sonlandırır
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50),

    # ModelCheckpoint: Sadece en iyi modeli kaydet (val_loss izleyerek)
    tf.keras.callbacks.ModelCheckpoint(filepath='eniyi_model.h5', verbose=1, save_best_only=True),

    # TensorBoard: Eğitim sürecinin ilerlemesini izlemek için logları kaydeder
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]



print(x_train.shape," ",y_train.shape)
history = model.fit(x_train_sk,y_train_sk,batch_size=1024, validation_data = (x_test_sk, y_test_sk),epochs=500,callbacks=my_callbacks)
model.summary()
train_score = model.evaluate(x_train_sk,y_train_sk,verbose=0)
print(train_score)
train_score = model.evaluate(x_test_sk,y_test_sk,verbose=0)
print(train_score)


plot_model(model, to_file='model.png',show_shapes=True)


test_tahminleri = model.predict(x_test_sk)


Df_gercek = pd.DataFrame(y_test_sk,columns=["Gerçek Y"])

Df_tahmin = pd.DataFrame(test_tahminleri,columns=["Tahmin Y"])


tahminDf = pd.concat([Df_gercek,Df_tahmin],axis=1)



#%%

sbn.scatterplot(x="Tahmin Y",y="Tahmin Y",data = tahminDf)

from tensorflow.keras.models import load_model 
model.save(r"son_yangin_tahmin_modeli.h5")


#%%
model_kaybim = pd.DataFrame(history.history)

# İlk figür: Kayıp ve doğrulama kaybı
plt.figure()
plt.plot(model_kaybim["loss"], label="Loss")
plt.plot(model_kaybim["val_loss"], label="val_loss")
plt.title("loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# İkinci figür: Doğruluk (accuracy)
plt.figure()
plt.plot(model_kaybim["accuracy"], label=" (Accuracy)")
plt.title("accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
