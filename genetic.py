import tensorflow as tf
import model as m
import random
from random import randint
import data as d
import time
import copy
from tensorflow.keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl

batch_size = 32

print("Loading Dataset....")
#Load cifar10
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)

#Load FER2013
x_train, y_train, x_test, y_test = d.FERLoad()

#Load ExpW
#x_train, y_train, x_test, y_test = d.ExpwLoad('origin', 0.1)
print("Dataset Loaded")

def individual(pos):
    return m.getParam()[pos]


def population(count, pos):
    pop = []
    alpha=1
    depth_multiplier=1
    activation=relu
    use_bias=True
    dropout=0.001

    optimizer=Nadam
    kernel_regularizer=[None]
    bias_regularizer=[None]
    activity_regularizer=[None]

    for i in range(count):
        pooling=AveragePooling2D()
        tmp = [alpha, depth_multiplier, activation, use_bias, dropout, pooling, optimizer, kernel_regularizer, bias_regularizer, activity_regularizer]
        tmp[pos] = m.getParam(pos)
        pop.append(copy.deepcopy(tmp))
    return pop

def fitness(individual, extra = 0):
    f = open("log.txt", "a")
    epochs = 1 + extra
    print("Alpha,", individual[0], "Depth_multiplier,", individual[1], "Activation,", individual[2],
        "Use_bias,", individual[3], "Dropout,", individual[4], "Pooling,", individual[5], "Optimizer,", individual[6],
        "Kernel_regularizer,", individual[7], "Bias_regularizer,", individual[8], "Activity_regularizer,", individual[9])
    f.write("Alpha,"+str(individual[0])+","+"Depth_multiplier,"+str(individual[1])+","+"Activation,"+str(individual[2])+","+
        "Use_bias,"+str(individual[3])+","+"Dropout,"+str(individual[4])+","+"Pooling,"+str(individual[5])+","+
        "Optimizer,"+str(individual[6])+","+"Kernel_regularizer,"+str(individual[7])+","+"Bias_regularizer,"+str(individual[8])+","+
        "Activity_regularizer,"+str(individual[9])+"\n")
    model = m.MobileNet(individual[0], individual[1], individual[2], individual[3], individual[4], individual[5], individual[6], individual[7], individual[8], individual[9]).model(input_shape=x_train[0].shape)
    start = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=epochs, batch_size=batch_size)
    end = time.time()
    cost = (end - start)/epochs
    f.write(str(history.history['val_accuracy'][-1])+','+str(cost)+"\n")
    #f.write(str(history.history['val_accuracy'][-1])+"\n")
    f.close()
    tf.keras.backend.clear_session()
    score = history.history['val_accuracy'][-1] - cost/1000
    return score

def grade(pop, extra = 0):
    summed = sum([fitness(x, extra) for x in pop])
    return summed / (len(pop) * 1.0)

def evolve(pop, pos, ln, retain=0.2, random_select=0.05, mutate=0.01, extra = 0,):
    fit = [ fitness(x, extra) for x in pop]
    graded = []
    for i in range(len(fit)):
        graded.append((fit[i], pop[i]))
    # graded = [ (fitness(x, target), x) for x in pop]
    graded = [x[1] for x in sorted(graded, reverse = True)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)

    # mutate some individuals
    for individual in parents:
        if mutate > random.random():
            individual = m.getParam()

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            if ln != 1:
                half = int(len(male) / 2)
            else:
                half = pos
            child = male[:half] + female[half:]
            children.append(child)

    parents.extend(children)

    return parents, sum(fit) / (len(pop) * 1.0)

def getParamRange(pop, pos):
    ln = int(len(pop) / 4)
    tmp = pop[:ln]
    tmp2 = []
    for i in range(len(tmp)):
        tmp2.append(tmp[i][pos])
    if type(tmp[0][pos])==type(1) or type(tmp[0][pos])==type(1.0):
        return [min(tmp2), max(tmp2)]
    else:
        return tmp2

def individual2d(alpha_r, depth_multiplier_r, activation_r, use_bias_r, dropout_r, pooling_r, optimizer_r, kernel_regularizer_r, bias_regularizer_r, activity_regularizer_r):
    alpha = random.uniform(alpha_r[0], alpha_r[1])
    depth_multiplier = random.choice(range(depth_multiplier_r[0],depth_multiplier_r[1]+1))
    dropout = random.uniform(dropout_r[0], dropout_r[1])
    activation = random.choice(activation_r)
    use_bias = random.choice(use_bias_r)
    pooling = random.choice(pooling_r)
    optimizer = random.choice(optimizer_r)
    kernel_regularizer = random.choice(kernel_regularizer_r)
    bias_regularizer = random.choice(bias_regularizer_r)
    activity_regularizer = random.choice(activity_regularizer_r)
    return [alpha, depth_multiplier, activation, use_bias, dropout, pooling, optimizer, kernel_regularizer, bias_regularizer, activity_regularizer]


def population2d(count, alpha_r, depth_multiplier_r, activation_r, use_bias_r, dropout_r, pooling_r, optimizer_r, kernel_regularizer_r, bias_regularizer_r, activity_regularizer_r):
    return [individual2d(alpha_r, depth_multiplier_r, activation_r, use_bias_r, dropout_r, pooling_r, optimizer_r, kernel_regularizer_r, bias_regularizer_r, activity_regularizer_r) for x in range(count)]
