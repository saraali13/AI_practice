
from sklearn.neural_network import MLPClassifier

ann = MLPClassifier(
    hidden_layer_sizes=(64,32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500
)

ann.fit(X_train, y_train)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

model = Sequential()

# Input layer
model.add(Dense(units=20,input_dim=20, activation='relu'))

# Hidden layers
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.3))#wot be included while training  drop 30%

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fit
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    batch_size=32,
                    epochs=50,
                   validation_data=(x_test,y_test))

model = Sequential([
    Dense(32, activation='relu', input_shape=(20,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.add(Dense(units=64, activation='sigmoid'), kernel_regularizer=l2(0.01))

#Activation Functions
#sigmoid,tanh,ReLU,leaky ReLU, Softmax

#loss funct
#regression-> MSE,MAE
#classification(bin)->binary cross entropy
#classification(multi)->categorical cross entropy
#classification(multi+int)->sparse categorical cross entropy

pred=model.predict(x_test)
pred_bin=np.round(pred)
acc=accuracy_score(y_test,pred_bin)

#for regression
model = Sequential()

# Input layer
model.add(Dense(units=20,input_dim=20, activation='relu'))

# Hidden layers
model.add(Dense(units=16, activation='relu'))

# Output layer
model.add(Dense(units=1, activation='linear'))

# Compile
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Fit
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    batch_size=32,
                    epochs=50,
                   validation_data=(x_test,y_test))

pred=model.predict(x_test)
pred_bin=np.round(pred)
acc=mean_square_error(y_test,pred)
