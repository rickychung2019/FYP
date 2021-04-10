import genetic as g
from tensorflow.keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl


import time
alpha_list = [1.25, 1.5, 1.75, 2.25, 2.75, 3.0]
depth_multiplier_list = [2,3]
activation_list = [relu, sigmoid, tanh, selu, elu]
bias_list = [False]
dropout_list = [0.2, 0.3, 0.5]
pooling_list = [None,  GlobalMaxPooling2D(), MaxPooling2D()]
optimizer_list = [RMSprop, Adam, Adagrad, Ftrl]
kernel_regularizer_list =[l1,l2]
bias_regularizer_list =[None,l2]
activation_regularizer_list =[l2,l1_l2]

pop2d = 100
numOfGen1d = 1
numOfGen2d = 4
numOfParam = 10


fitness_history = []
paramRange2 = [alpha_list, depth_multiplier_list, activation_list, bias_list, dropout_list, pooling_list, optimizer_list, kernel_regularizer_list, bias_regularizer_list, activation_regularizer_list]
# 2d genetic
p2 = g.population2d(pop2d, paramRange2[0], paramRange2[1], paramRange2[2], paramRange2[3], paramRange2[4],paramRange2[5], paramRange2[6], paramRange2[7], paramRange2[8], paramRange2[9])
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
    extra = int(i)
    p, tmp = g.evolve(p2, 0, extra = extra)
    fitness_history.append(tmp)

print("#############################################################")
print("#############################################################")
print("#############################################################")
print("Grading the Final Population")
f = open("log.txt", "a")
f.write("Grading the Final Population\n")
f.close()
fitness_history.append(g.grade(p, int(numOfGen2d/2)))
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
