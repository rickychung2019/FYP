import tensorflow as tf
import random
from random import randint
from tensorflow.keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl

random.seed(1)

def getParam():
    alpha = random.choice([i*0.25 for i in range(1, 13)]) #12 choices
    depth_multiplier = random.choice([1,2,3]) #3 choices
    activation = random.choice([relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential]) #9 choices
    bias = random.choice([False, True]) # 2 choices
    dropout = random.choice([i*0.1 for i in range(0, 6)]) # 6
    pooling = random.choice([None, AveragePooling2D(), GlobalAveragePooling2D(), GlobalMaxPooling2D(), MaxPooling2D()]) # 5 choices
    optimizer = random.choice([SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl]) # 8 choices
    kernel_regularizer = random.choice([None,l1,l2,l1_l2]) # 3 choices
    bias_regularizer = random.choice([None,l1,l2,l1_l2]) # 3 choices
    activity_regularizer = random.choice([None,l1,l2,l1_l2]) # 3 choices

    return [alpha, depth_multiplier, activation, bias, dropout, pooling, optimizer, kernel_regularizer, bias_regularizer, activity_regularizer]

class MobileNet(tf.keras.Model):
    def __init__(self, alpha=1, depth_multiplier=1, activation=relu, use_bias=True, dropout=0.001, pooling=AveragePooling2D(), optimizer=Nadam, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):
        super(MobileNet, self).__init__()
        self.alpha = alpha
        self.depth_multiplier = depth_multiplier
        self.activation = activation
        self.use_bias = use_bias
        self.pooling = pooling
        self.dropoutRate = dropout
        self.optimizer = optimizer
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

        self.dropout = Dropout(self.dropoutRate, name='dropout')
        self.conv2d_1 = Conv2D(int(32*alpha), (3,3), strides=2, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_1_1 = DepthwiseConv2D((3,3), strides=1, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_1_2 = DepthwiseConv2D((3,3), strides=1, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_1_3 = DepthwiseConv2D((3,3), strides=1, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_1_4_1 = DepthwiseConv2D((3,3), strides=1, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_1_4_2 = DepthwiseConv2D((3,3), strides=1, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_1_4_3 = DepthwiseConv2D((3,3), strides=1, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_1_4_4 = DepthwiseConv2D((3,3), strides=1, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_1_4_5 = DepthwiseConv2D((3,3), strides=1, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_1_5 = DepthwiseConv2D((3,3), strides=1, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_2_1 = DepthwiseConv2D((3,3), strides=2, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_2_2 = DepthwiseConv2D((3,3), strides=2, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_2_3 = DepthwiseConv2D((3,3), strides=2, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.dwconv_2_4 = DepthwiseConv2D((3,3), strides=2, padding="same", depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, depthwise_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)

        self.pwconv_1 = Conv2D(int(64*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_2_1 = Conv2D(int(128*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_2_2 = Conv2D(int(128*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_3_1 = Conv2D(int(256*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_3_2 = Conv2D(int(256*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_4_1 = Conv2D(int(512*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_4_2_1 = Conv2D(int(512*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_4_2_2 = Conv2D(int(512*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_4_2_3 = Conv2D(int(512*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_4_2_4 = Conv2D(int(512*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_4_2_5 = Conv2D(int(512*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_5_1 = Conv2D(int(1024*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.pwconv_5_2 = Conv2D(int(1024*alpha), (1,1), strides=1, padding="same", use_bias=self.use_bias, kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer, activity_regularizer=self.activity_regularizer)
        self.flatten = Flatten(name='flatten')
        self.dense = Dense (7, activation='softmax')

    def call(self, inputs):
        #standard 2D convolution
        x = self.conv2d_1(inputs)
        x = BatchNormalization()(x)
        x = self.activation(x)

        #depth wise separable convolutions
        x = self.depthwiseSeparableConvs_1(x)
        x = self.depthwiseSeparableConvs_2(x)
        x = self.depthwiseSeparableConvs_3(x)

        #pooling
        if self.pooling is None:
            pass
        else:
            x = self.pooling(x)

        #Dropout
        x = self.dropout(x)

        #Fully connected layer
        x = self.flatten(x)

        #Output layer
        x = self.dense(x)

        return x

    def model(self, input_shape=(224,224,3)):
        x = Input(shape=input_shape)
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        for layer in model.layers:
            layer.trainable=True
        model.compile(
          optimizer=self.optimizer(0.01),
          loss='categorical_crossentropy',
          metrics=['accuracy'],
        )
        return model

    def depthwiseSeparableConvs_1(self,x):
        x = self.dwconv_1_1(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_1(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_2_1(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_2_1(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_1_2(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_2_2(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_2_2(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_3_1(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_1_3(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_3_2(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_2_3(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_4_1(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        return x
    def depthwiseSeparableConvs_2(self,x):
        x = self.dwconv_1_4_1(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_4_2_1(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_1_4_2(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_4_2_2(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_1_4_3(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_4_2_3(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_1_4_4(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_4_2_4(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_1_4_5(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_4_2_5(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        return x

    def depthwiseSeparableConvs_3(self,x):
        x = self.dwconv_2_4(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_5_1(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        x = self.dwconv_1_5(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        x = self.pwconv_5_2(x)
        x = BatchNormalization()(x)
        x = self.activation(x)

        return x
