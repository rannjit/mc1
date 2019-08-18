#coding=utf-8

try:
    import pandas as pd
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    from keras import backend as K
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers import LSTM
except:
    pass

try:
    from keras.layers.core import Dense, Dropout
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

X_train = X_train
X_test = X_test
y_train = y_train
y_test = y_test    


def keras_fmin_fnct(space):

    
    model = Sequential()
    
    model.add(LSTM(units=space['units'], activation=space['activation'],
                   kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)),
                   input_shape=(timesteps, input_dim),return_sequences=True)
    model.add(Dropout(space['Dropout'])) # Adding a dropout layer
    #Adding a second LSTM Layer
    model.add(LSTM(units=space['units_1'], activation=space['activation_1'],
                   kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)),
                   input_shape=(timesteps, input_dim))
    model.add(BatchNormalization()) # Adding batch normalization     
    model.add(Dropout(space['Dropout_1'])) # Adding a dropout layer
    model.add(Dense(n_classes, activation='sigmoid')) # Adding a dense output layer with sigmoid activation    
    
    # Tune the optimzers
    adam = keras.optimizers.Adam(lr=space['lr'])
    rmsprop = keras.optimizers.RMSprop(lr=space['lr_1'])
    sgd = keras.optimizers.SGD(lr=space['lr_2'])
    
    choiceval = space['choiceval']
    if choiceval == 'adam':
        optimizer = adam
    elif choiceval == 'rmsprop':
        optimizer = rmsprop
    else:
        optimizer = sgd
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) # Compiling the model
    model.fit(X_train, y_train, batch_size=space['batch_size'], validation_data=(X_test, y_test), epochs=epochs) # Training the model
    #history=model.history 
    
    score, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'units': hp.choice('units', [32,48,64,80]),
        'activation': hp.choice('activation', ['tanh','relu','sigmoid','softplus']),
        'Dropout': hp.uniform('Dropout', 0, 1),
        'units_1': hp.choice('units_1', [32,48,64,80]),
        'activation_1': hp.choice('activation_1', ['tanh','relu','sigmoid','softplus']),
        'Dropout_1': hp.uniform('Dropout_1', 0, 1),
        'lr': hp.choice('lr', [10**-4,10**-3, 10**-2]),
        'lr_1': hp.choice('lr_1', [10**-4,10**-3, 10**-2, 10**-1]),
        'lr_2': hp.choice('lr_2', [10**-4,10**-3, 10**-2]),
        'choiceval': hp.choice('choiceval', ['adam', 'sgd', 'rmsprop']),
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
    }
