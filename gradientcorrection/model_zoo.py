################################################################################### CNN-local and CNN-dist
from tensorflow import keras
from gradientcorrection import layers, utils

#Default models 
def cnn_dist_model(kernel_size=19, activation='relu'):
        
    # input layer
    inputs = keras.layers.Input(shape=(200,4))

    # block 1
    nn = layers.conv_layer(inputs,
                           num_filters=24, 
                           kernel_size=kernel_size, 
                           padding='same', 
                           activation=activation, 
                           dropout=0.1,
                           l2=1e-6, 
                           bn=True)

    # layer 2
    nn = layers.conv_layer(nn,
                           num_filters=32, 
                           kernel_size=7, 
                           padding='same', 
                           activation='relu', 
                           dropout=0.2,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 3
    nn = layers.conv_layer(nn,
                           num_filters=48, 
                           kernel_size=7, 
                           padding='valid', 
                           activation='relu', 
                           dropout=0.3,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 4
    nn = layers.conv_layer(nn,
                           num_filters=64, 
                           kernel_size=3, 
                           padding='valid', 
                           activation='relu', 
                           dropout=0.4,
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=3, 
                                strides=3, 
                                padding='same'
                                )(nn)

    # layer 5
    nn = keras.layers.Flatten()(nn)
    nn = layers.dense_layer(nn, num_units=96, activation='relu', 
                            dropout=0.5, l2=1e-6, bn=True)

    # Output layer 
    logits = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model




def cnn_local_model(kernel_size=19, activation='relu'):
      
    # input layer
    inputs = keras.layers.Input(shape=(200,4))

    # layer 1
    nn = layers.conv_layer(inputs,
                           num_filters=24, 
                           kernel_size=kernel_size, 
                           padding='same', 
                           activation=activation, 
                           dropout=0.1,
                           l2=1e-6,
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=50)(nn)

    # layer 2
    nn = layers.conv_layer(nn, 
                           num_filters=48, 
                           kernel_size=3, 
                           padding='same',
                           activation='relu', 
                           dropout=0.2, 
                           l2=1e-6, 
                           bn=True)
    nn = keras.layers.MaxPool1D(pool_size=2)(nn)

    # layer 3
    nn = keras.layers.Flatten()(nn)
    nn = layers.dense_layer(nn, num_units=96, activation='relu', 
                            dropout=0.5, l2=1e-6, bn=True)

    # Output layer 
    logits = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


############################################################  Similar CNN models, but with the initializer. 

def cnn_dist_init_model(activation='relu', sigma=0.05):
        
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=sigma)

    # input layer
    inputs = keras.layers.Input(shape=(200,4))

    # block 1
    nn = conv_layer(inputs,
                    num_filters=24, 
                    kernel_size=19, 
                    padding='same', 
                    activation=activation, 
                    kernel_initializer=initializer,
                    dropout=0.1,
                    l2=1e-6, 
                    bn=True)

    # layer 2
    nn = conv_layer(nn,
                    num_filters=32, 
                    kernel_size=7, 
                    padding='same', 
                    activation='relu', 
                    dropout=0.2,
                    kernel_initializer=initializer,
                    l2=1e-6, 
                    bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 3
    nn = conv_layer(nn,
                    num_filters=48, 
                    kernel_size=7, 
                    padding='valid', 
                    activation='relu', 
                    kernel_initializer=initializer,
                    dropout=0.3,
                    l2=1e-6, 
                    bn=True)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 4
    nn = conv_layer(nn,
                    num_filters=64, 
                    kernel_size=3, 
                    padding='valid', 
                    activation='relu', 
                    kernel_initializer=initializer,
                    dropout=0.4,
                    l2=1e-6, 
                    bn=True)
    nn = keras.layers.MaxPool1D(pool_size=3, 
                                strides=3, 
                                padding='same'
                                )(nn)

    # layer 5
    nn = keras.layers.Flatten()(nn)
    nn = dense_layer(nn, num_units=96, activation='relu', 
                     kernel_initializer=initializer,
                     dropout=0.5, l2=1e-6, bn=True)

    # Output layer 
    logits = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)
        
    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model




def cnn_local_init_model(activation='relu', sigma=0.05):
      
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=sigma)
    
    # input layer
    inputs = keras.layers.Input(shape=(200,4))

    # layer 1
    nn = conv_layer(inputs,
                    num_filters=24, 
                    kernel_size=19, 
                    padding='same', 
                    activation=activation, 
                    kernel_initializer=initializer,
                    dropout=0.1,
                    l2=1e-6,
                    bn=True)
    nn = keras.layers.MaxPool1D(pool_size=50)(nn)

    # layer 2
    nn = conv_layer(nn, 
                    num_filters=48, 
                    kernel_size=3, 
                    padding='same',
                    activation='relu', 
                    kernel_initializer=initializer,
                    dropout=0.2, 
                    l2=1e-6, 
                    bn=True)
    nn = keras.layers.MaxPool1D(pool_size=2)(nn)

    # layer 3
    nn = keras.layers.Flatten()(nn)
    nn = dense_layer(nn, num_units=96, activation='relu', 
                    kernel_initializer=initializer,
                    dropout=0.5, l2=1e-6, bn=True)

    # Output layer 
    logits = keras.layers.Dense(1, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
