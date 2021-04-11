import genetic as g
from tensorflow.keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl


import time
alpha_list = [i*0.25 for i in range(1, 12+1)] #12 choices
depth_multiplier_list = [1,2,3] #3 choices
activation_list = [relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential] #9 choices
bias_list = [False, True] # 2 choices
dropout_list = [i*0.05 for i in range(0, 11)] # 6 choices 0, 0.1,..., 0.5
pooling_list = [None, GlobalAveragePooling2D(), GlobalMaxPooling2D()] # 5 choices
optimizer_list = [SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl] # 8 choices
kernel_regularizer_list =[None,l1_l2] # 3 choices
bias_regularizer_list =[None,l1_l2] # 3 choices
activation_regularizer_list =[None,l1_l2] # 3 choices
layer_list = [1,2,3,4,5,6,7,8,9,10] #10 choices

pop2d = 100
numOfGen1d = 1
numOfGen2d = 4


fitness_history = []
paramRange2 = [alpha_list, depth_multiplier_list, activation_list, bias_list, dropout_list, pooling_list, optimizer_list, kernel_regularizer_list, bias_regularizer_list, activation_regularizer_list,layer_list]
start = time.time()
# 2d genetic
p2 = g.population2d(pop2d, paramRange2[0], paramRange2[1], paramRange2[2], paramRange2[3], paramRange2[4],paramRange2[5], paramRange2[6], paramRange2[7], paramRange2[8], paramRange2[9],paramRange2[10])
for i in range(1, numOfGen2d+1):
    print("#########################################################")
    print("#############################################################")
    print("#############################################################")
    print(str(i) + "EVOLUTION")
    print("#########################################################")
    print("#############################################################")
    print("#############################################################")
    f = open("log.txt", "a")
    f.write(str(i)+"Evolution\n")
    f.close()
    extra = 2+int(i/2)
    p, tmp = g.evolve(p2, 1, extra = extra)
    fitness_history.append(tmp)

print("#############################################################")
print("#############################################################")
print("#############################################################")
print("Grading the Final Population")
f = open("log.txt", "a")
f.write("Grading the Final Population\n")
f.close()
fitness_history.append(g.grade(p, 2+int(numOfGen2d/2)))
print("#############################################################")
print("#############################################################")
print("#############################################################")

end = time.time()

for datum in fitness_history:
    print(datum)

f = open("log.txt", "a")
cost = end - start
f.write("Time Spent\n")
f.wrtie(str(cost))
f.close()
