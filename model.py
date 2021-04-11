import tensorflow as tf
import random
from random import randint
from tensorflow.keras.layers import ZeroPadding2D, Input, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl

random.seed(1)

def getParam():
    alpha = random.choice([i*0.25 for i in range(1, 13)]) #12 choices
    depth_multiplier = random.choice([1,2,3]) #3 choices
    activation = random.choice([relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential]) #9 choices
    bias = random.choice([False, True]) # 2 choices
    dropout = random.choice([i*0.05 for i in range(0, 11)]) # 11
    pooling = random.choice([None, GlobalAveragePooling2D(), GlobalMaxPooling2D()]) # 3 choices
    optimizer = random.choice([SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl]) # 8 choices
    kernel_regularizer = random.choice([None,l1,l2]) # 3 choices
    bias_regularizer = random.choice([None,l1,l2]) # 3 choices
    activity_regularizer = random.choice([None,l1,l2]) # 3 choices
    layer = random.choice([1,2,3,4,5,6,7,8,9,10]) #5 choices

    return [alpha, depth_multiplier, activation, bias, dropout, pooling, optimizer, kernel_regularizer, bias_regularizer, activity_regularizer, layer]

class MobileNet(object):
    def __init__(self, alpha=1, depth_multiplier=1, activation=relu, use_bias=True, dropout=0.0, pooling=GlobalAveragePooling2D(), optimizer=RMSprop, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, layer=5):
        super(MobileNet, self).__init__()
        self.alpha = alpha
        self.depth_multiplier = depth_multiplier
        self.activation = activation
        self.use_bias = use_bias
        self.pooling = pooling
        self.dropoutRate = dropout
        self.optimizer = optimizer
        self.layer = layer
        if kernel_regularizer is None:
            self.kernel_regularizer = None
        else:
            self.kernel_regularizer = kernel_regularizer()
        if bias_regularizer is None:
            self.bias_regularizer = None
        else:
            self.bias_regularizer = bias_regularizer()
        if activity_regularizer is None:
            self.activity_regularizer = None
        else:
            self.activity_regularizer = activity_regularizer()

    def standard2DConv(self,inputs, filters):
        scaled = int(filters * self.alpha)
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(inputs)
        x = Conv2D(scaled, 3,strides=2,padding='valid', use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)(x)
        x = BatchNormalization(axis=-1)(x)
        return self.activation(x)

    def dwSConv(self,inputs, filters,strides):
        scaled = int(filters * self.alpha)
        if strides == 1:
            padding='same'
            x = inputs
        else:
            padding='valid'
            x = ZeroPadding2D(((0, 1), (0, 1)))(inputs)
        x = DepthwiseConv2D(3,padding=padding,strides=strides,depth_multiplier=self.depth_multiplier,use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = Conv2D(scaled, 1, strides=1, padding='same',use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)(x)
        x = BatchNormalization()(x)
        return self.activation(x)

    def model(self, input_shape):
        input_layer = Input(shape=input_shape)
        x = self.standard2DConv(input_layer, 32)
        x = self.dwSConv(x,64,1)
        x = self.dwSConv(x,128,2)
        x = self.dwSConv(x,128,1)
        x = self.dwSConv(x,256,2)
        x = self.dwSConv(x,256,1)
        x = self.dwSConv(x,512,2)
        for i in range(self.layer):
            x = self.dwSConv(x,512,1)
        x = self.dwSConv(x,1024,2)
        x = self.dwSConv(x,1024,1)

        if self.pooling is not None:
            x = self.pooling(x)
        if self.dropoutRate != 0:
            x = Dropout(self.dropoutRate, name='dropout')(x)
        x = Flatten(name='flatten')(x)
        output_layer = Dense(10, activation='softmax')(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        for layer in model.layers:
            layer.trainable=True
        model.compile(
          optimizer=self.optimizer(0.01),
          loss='categorical_crossentropy',
          metrics=['accuracy'],
        )
        return model
