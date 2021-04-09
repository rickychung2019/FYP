import genetic as g

# modify setting here
pop1d = 50
pop2d = 200
numOfGen1d = 1
numOfGen2d = 4
numOfParam = 10


fitness_history = []
paramRange = []

# 1d genetic
for i in range(numOfParam):
    p = g.population(pop1d, i)
    for j in range(1, numOfGen1d+1):
        print("#########################################################")
        print("#########################################################")
        print("#########################################################")
        print(str(j) + "EVOLUTION" + " FOR PARAM " + str(i))
        print("#########################################################")
        print("#########################################################")
        print("#########################################################")
        f = open("log.txt", "a")
        f.write(str(j) + "EVOLUTION" + " FOR PARAM " + str(i)+"\n")
        f.close()
        p, tmp = g.evolve(p, i, 1)
    paramRange.append(g.getParamRange(p, i))


f = open("log.txt", "a")
f.write("ParamRange,"+str(paramRange[0])+","+str(paramRange[1])+","+str(paramRange[2])+","+str(paramRange[3])+","+str(paramRange[4])+","+
    str(paramRange[5])+","+str(paramRange[6])+","+str(paramRange[7])+","+str(paramRange[8])+","+str(paramRange[9])+"\n")
f.close()

# 2d genetic
p2 = g.population2d(pop2d, paramRange[0], paramRange[1], paramRange[2], paramRange[3], paramRange[4],paramRange[5], paramRange[6], paramRange[7], paramRange[8], paramRange[9])
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
    p, tmp = g.evolve(p2, 0, numOfParam, extra = extra)
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

for datum in fitness_history:
    print(datum)
