from tensorflow import keras


def cnn_deep(input_shape, output_shape, activation='relu', num_filters=24, initializer='he_normal'):
        
    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1 
    nn = keras.layers.Conv1D(filters=num_filters, kernel_size=11, padding='same', 
                             kernel_regularizer=1e-6, kernel_initializer=initializer)(inputs)     
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    # layer 2
    nn = keras.layers.Conv1D(filters=32, kernel_size=7, padding='same', 
                             kernel_regularizer=1e-6, kernel_initializer=initializer)(inputs)     
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 3
    nn = keras.layers.Conv1D(filters=48, kernel_size=7, padding='same', 
                             kernel_regularizer=1e-6, kernel_initializer=initializer)(inputs)     
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 4
    nn = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', 
                             kernel_regularizer=1e-6, kernel_initializer=initializer)(inputs)     
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.4)(nn)
    nn = keras.layers.MaxPool1D(pool_size=3)(nn)

    # layer 5
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(96, kernel_regularizer=1e-6, kernel_initializer=initializer)(inputs)     
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # Output layer 
    logits = keras.layers.Dense(output_shape, activation='linear')(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    return keras.Model(inputs=inputs, outputs=outputs)




def cnn_shallow(input_shape, output_shape, activation='relu', num_filters=24, initializer='he_normal'):
   
    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1 
    nn = keras.layers.Conv1D(filters=num_filters, kernel_size=19, padding='same', 
                             kernel_regularizer=1e-6, kernel_initializer=initializer)(inputs)     
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Dropout(0.1)(nn)
    nn = keras.layers.MaxPool1D(pool_size=50)(nn)

    # layer 2
    nn = keras.layers.Conv1D(filters=48, kernel_size=3, padding='same', 
                             kernel_regularizer=1e-6, kernel_initializer=initializer)(inputs)     
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = keras.layers.MaxPool1D(pool_size=2)(nn)

    # layer 3
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(96, kernel_regularizer=1e-6, kernel_initializer=initializer)(inputs)     
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # Output layer 
    logits = keras.layers.Dense(output_shape, activation='linear')(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    return keras.Model(inputs=inputs, outputs=outputs)

  