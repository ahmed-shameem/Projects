import numpy as np
import random
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math,time,sys
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from functools import partial
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

Ngen = 30                # Number of generations
N  = 5                   # MxN: reef size
M  = 5                    # MxN: reef size
Fb = 0.6                   # Broadcast prob.
Fa = 0.4                   # Asexual reproduction prob.
Fd = 0.1                   # Fraction of the corals to be eliminated in the depredation operator.
r0 = 0.6                   # Free/total initial proportion
k  = 3                     # Number of opportunities for a new coral to settle in the reef
Pd = 0.1                   # Depredation prob.
omega = 0.9
ke = 0.2


def initialise(L):
    O = int(np.round(N*M*r0)) # number of occupied reefs
    A = np.random.randint(2, size=[O, L]) #size=(O, L), use tuple of ints
    B = np.zeros([((N*M)-O), L], int)
    REEFpob = np.concatenate([A, B]) # Population creation
    REEF = np.array((REEFpob.any(axis=1)),int)
    return (REEF, REEFpob)
# print(REEFpob)
# print(REEF)

def fitness(agent, trainX, testX, trainy, testy):
    # print(agent)
    cols=np.flatnonzero(agent)
    # print(cols)
    val=1
    if np.shape(cols)[0]==0:
        return val
    clf=KNeighborsClassifier(n_neighbors=5)
    #clf=MLPClassifier(alpha=0.001, hidden_layer_sizes=(1000,500,100),max_iter=2000,random_state=4)
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=1-clf.score(test_data,testy)

    #in case of multi objective  []
    set_cnt=sum(agent)
    set_cnt=set_cnt/np.shape(agent)[0]
    val=omega*val+(1-omega)*set_cnt
    return val

def test_accuracy(agent, trainX, testX, trainy, testy):
    cols=np.flatnonzero(agent)
    val=1
    if np.shape(cols)[0]==0:
        return val
    # clf = RandomForestClassifier(n_estimators=300)
    #clf=MLPClassifier(alpha=0.001, hidden_layer_sizes=(1000,500,100),max_iter=2000,random_state=4)
    clf=KNeighborsClassifier(n_neighbors=5)
    # clf=MLPClassifier( alpha=0.01, max_iterno=1000) #hidden_layer_sizes=(1000,500,100)
    #cross=4
    #test_size=(1/cross)
    #X_train, X_test, y_train, y_test = train_test_split(trainX, trainy,  stratify=trainy,test_size=test_size)
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=clf.score(test_data,testy)
    return val

def onecnt(agent):
    return sum(agent)

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def broadcastspawning(REEF, REEFpob, L):
    # get  number of spawners, forzed to be even (pairs of corals to create one larva)
    #np.random.seed(time.time()*10*98526+214)
    nspawners = int(np.round(Fb*np.sum(REEF)))

    if (nspawners%2) !=0:
        nspawners=nspawners-1

    # get spawners and divide them in two groups
    p = np.where(REEF!=0)[0]
    a = np.random.permutation(p)
    spawners = a[:nspawners]
    spawners1 = REEFpob[spawners[0:int(nspawners/2)], :]
    spawners2 = REEFpob[spawners[int(nspawners/2):], :]

    # get crossover mask for some of the methods below
    (a,b) = spawners1.shape
    mask = np.random.randint(2, size=[a,b])

    # all zeros and all ones doesn't make sense, Not produces crossover
    pos = np.where(np.sum(mask, axis= 1)==0)[0]
    mask[pos, np.random.randint(L, size=[len(pos)])] = 1
    pos = np.where(np.sum(mask, axis= 1)==1)[0]
    mask[pos, np.random.randint(L, size=[len(pos)])] = 0

    not_mask = np.logical_not(mask)

    ESlarvae1 = np.multiply(spawners1, not_mask) + np.multiply(spawners2, mask)
    ESlarvae2 = np.multiply(spawners2, not_mask) + np.multiply(spawners1, mask)
    ESlarvae = np.concatenate([ESlarvae1, ESlarvae2])
    return ESlarvae

def larvaemutation_function(brooders, pos):
    (nbrooders, _) = brooders.shape
    brooders[range(nbrooders), pos] = np.logical_not(brooders[range(nbrooders), pos])
    return (brooders)


def brooding(REEF, REEFpob):
    npolyps = 1

    # get the brooders
    #np.random.seed(time.time()*5*968+52*78)
    nbrooders= int(np.round((1-Fb)*np.sum((REEF))))

    p = np.where(REEF!=0)[0]
    a = np.random.permutation(p)
    brooders = a[0:nbrooders]
    brooders = REEFpob[brooders, :]

    pos = np.random.randint(brooders.shape[1], size=(npolyps, nbrooders))

    brooders = larvaemutation_function(brooders, pos)

    return brooders

def _settle_larvae(larvae, larvaefitness, REEF, REEFpob, REEFfitness, indices):
    REEF[indices] = 1

    j = 0
    for i in range(len(larvae)):
        REEFpob[indices[j]] = larvae[i]
        j = j+1
    #REEFpob[indices, :] = larvae

    i = 0
    for j in range(len(larvaefitness)):
        REEFfitness[indices[i]] = larvaefitness[j]
        i = i+1

    return REEF, REEFpob, REEFfitness

def larvaesettling(REEF, REEFpob, REEFfitness, larvae, larvaefitness, k):

    #np.random.seed(time.time()*523+69*87)
    nREEF = len(REEF)

    free = np.where(REEF==0)[0]

    larvae_emptycoral = larvae[:len(free), :]

    fitness_emptycoral = larvaefitness[:len(free)]

    REEF, REEFpob, REEFfitness = _settle_larvae(larvae_emptycoral, fitness_emptycoral, REEF, REEFpob, REEFfitness, free)

    larvae = larvae[len(free):, :]  # update larvae

    larvaefitness = larvaefitness[len(free):]

    for larva, larva_fitness in zip(larvae, larvaefitness):
        reef_indices = np.random.randint(nREEF, size=k)
        reef_index = reef_indices[0]
        fitness_comparison = []
        if not REEF[reef_index]: # empty coral
            REEFpob[reef_index] = larva
            REEFfitness[reef_index] = larva_fitness
            REEF[reef_index] = 1
        else:                  # occupied coral
            #fitness_comparison = larva_fitness < REEFfitness[reef_indices]
            for i in range(len(reef_indices)):
                fitness_comparison.append(larva_fitness < REEFfitness[i])
                i = i+1

            if np.any(fitness_comparison):
                reef_index = reef_indices[np.where(fitness_comparison)[0][0]]
                REEFpob[reef_index] = larva
                REEFfitness[reef_index] = larva_fitness
                #REEF, REEFpob, REEFfitness = _settle_larvae(larva, larva_fitness, REEF, REEFpob, REEFfitness, reef_index)

    return (REEF,REEFpob,REEFfitness)

def budding(REEF, REEFpob, fitness):
    pob = REEFpob[np.where(REEF==1)[0], :]
    #fitness = fitness[np.where(REEF==1)]
    N = pob.shape[0]
    NA = int(np.round(Fa*N))

    ind = np.argsort(fitness)

    #fitness = fitness[ind]
    fitness = np.sort(fitness)
    Alarvae = pob[ind[0:NA], :]
    Afitness = fitness[0:NA]
    return (Alarvae, Afitness)

def depredation(REEF, REEFpob, REEFfitness, empty_coral, empty_coral_fitness):
    #np.random.seed(time.time()*12+10*98+63)

    # Sort by worse fitness
    ind = np.argsort(REEFfitness)
    ind = ind[::-1]

    sortind = ind[:int(np.round(Fd*REEFpob.shape[0]))]
    p = np.random.rand(len(sortind))
    dep = np.where(p<Pd)[0]
    REEF[sortind[dep]] = 0
    REEFpob[sortind[dep], :] = empty_coral
    i = 0
    for j in range(len(dep)):
        REEFfitness[sortind[dep[j]]] = empty_coral_fitness[i]
        i = i+1

    #REEFfitness[sortind[dep]] = empty_coral_fitness
    return (REEF,REEFpob,REEFfitness)

def extremedepredation(REEF, REEFpob, REEFfitness, ke, empty_coral, empty_coral_fitness):
    (U, indices, count) = np.unique(REEFpob, return_index=True, return_counts=True, axis=0)
    if len(np.where(np.sum(U, axis= 1)==0)[0]) !=0:
        zero_ind = int(np.where(np.sum(U, axis= 1)==0)[0])
        indices = np.delete(indices, zero_ind)
        count = np.delete(count, zero_ind)

    while np.where(count>ke)[0].size>0:
        higherk = np.where(count>ke)[0]
        REEF[indices[higherk]] = 0
        REEFpob[indices[higherk], :] = empty_coral
        i = 0
        for j in range(len(higherk)):
            REEFfitness[indices[higherk[j]]] = empty_coral_fitness[i]
            i = i+1
        #REEFfitness[indices[higherk]] = empty_coral_fitness

        (U, indices, count) = np.unique(REEFpob, return_index=True, return_counts=True, axis=0)
        if len(np.where(np.sum(U, axis= 1)==0)[0]) !=0:
            zero_ind = int(np.where(np.sum(U, axis= 1)==0)[0])
            indices = np.delete(indices, zero_ind)
            count   = np.delete(count, zero_ind)

    return (REEF,REEFpob,REEFfitness)

def get_global_best(REEFpob, REEFfitness):
    best_fit = 200000
    i = 0
    for coral in REEFpob:
        if REEFfitness[i] < best_fit:
            best_agent = coral
            best_fit = REEFfitness[i]
        i = i+1

    return best_agent, best_fit

def randomwalk(agent):
    percent = 30
    percent /= 100
    neighbor = agent.copy()
    size = len(agent)
    upper = int(percent*size)
    if upper <= 1:
        upper = size
    x = random.randint(1,upper)
    pos = random.sample(range(0,size - 1),x)
    for i in pos:
        neighbor[i] = 1 - neighbor[i]
    return neighbor


def adaptiveBeta(agent, agentFit, trainX, testX, trainy, testy):
    bmin = 0.1 #parameter: (can be made 0.01)
    bmax = 1
    maxIter = 20 # parameter: (can be increased )

#     agentFit = agent[1]
#     agent = agent[0].copy()
    for curr in range(maxIter):
        neighbor = agent.copy()
        size = len(neighbor)
        neighbor = randomwalk(neighbor)

        beta = bmin + (curr / maxIter)*(bmax - bmin)
        for i in range(size):
            random.seed( time.time() + i )
            if random.random() <= beta:
                neighbor[i] = agent[i]
        neighFit = fitness(neighbor,trainX,testX,trainy,testy)
        if neighFit <= agentFit:
            agent = neighbor.copy()
            agentFit = neighFit



    return (agent,agentFit)


def coral_reef(dataset):
    df = pd.read_csv(dataset)
    a, b = np.shape(df)
    data = df.values[:,0:b-1]
    label = df.values[:,b-1]
    dimension = data.shape[1]

    cross = 5
    test_size = (1/cross)
    trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size, random_state=(7+17*int(time.time()%1000)))
    #clf=MLPClassifier(alpha=0.001, hidden_layer_sizes=(1000,500,100),max_iter=2000, random_state=4)
    clf=KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainX,trainy)
    val=clf.score(testX,testy)
    whole_accuracy = val
    print("Total Acc: ",val)

    #Reef initialization
    (REEF, REEFpob) = initialise(dimension)

    REEF_fitness = []
    for coral in REEFpob:
        coral_fitness = fitness(coral, trainX, testX, trainy, testy)
        REEF_fitness.append(coral_fitness)
    REEFfitness = deepcopy(REEF_fitness)

    cr_gbest, cr_gbestfit = get_global_best(REEFpob, REEFfitness)

    # Store empty coral and its fitness in an attribute for later use
    empty_coral_index = np.where(REEF == 0)[0][0]
    empty_coral = REEFpob[empty_coral_index, :].copy()
    EMPTY_CORAL_FITNESS = []
    for coral in empty_coral.reshape((1, len(empty_coral)))[0]:
        empty_coral_fit = fitness(coral, trainX, testX, trainy, testy)
        EMPTY_CORAL_FITNESS.append(empty_coral_fit)
    empty_coral_fitness = deepcopy(EMPTY_CORAL_FITNESS)


    for n in range(Ngen):
        ESlarvae = broadcastspawning(REEF, REEFpob, dimension)
        ISlarvae = brooding(REEF, REEFpob)

        # larvae fitness
        ESfitness_util = []
        ISfitness_util = []
        for coral in ESlarvae:
            fit_ind = fitness(coral, trainX, testX, trainy, testy)
            ESfitness_util.append(fit_ind)
        ESfitness = deepcopy(ESfitness_util)
        for coral in ISlarvae:
            fit_ind = fitness(coral, trainX, testX, trainy, testy)
            ISfitness_util.append(fit_ind)
        ISfitness = deepcopy(ISfitness_util)

        # Larvae setting
        larvae = np.concatenate([ESlarvae,ISlarvae])
        larvaefitness = np.concatenate([ESfitness, ISfitness])
        (REEF, REEFpob, REEFfitness) = larvaesettling(REEF, REEFpob, REEFfitness, larvae, larvaefitness, k)


        # Asexual reproduction
        (Alarvae, Afitness) = budding(REEF, REEFpob, REEFfitness)
        (REEF, REEFpob, REEFfitness) = larvaesettling(REEF, REEFpob, REEFfitness, Alarvae, Afitness, k)


        if n!=Ngen:
            (REEF, REEFpob, REEFfitness) = depredation(REEF, REEFpob, REEFfitness, empty_coral, empty_coral_fitness)
            (REEF, REEFpob, REEFfitness) = extremedepredation(REEF, REEFpob, REEFfitness, int(np.round(ke*N*M)), empty_coral, empty_coral_fitness)

        #Adaptive BETA
        for i in range(len(REEFpob)):
            REEFpob[i], REEFfitness[i] = adaptiveBeta(REEFpob[i], REEFfitness[i], trainX, testX, trainy, testy)

        cr_currbest, cr_currbestfit = get_global_best(REEFpob, REEFfitness)
        if cr_currbestfit < cr_gbestfit:
            cr_gbest = cr_currbest
            cr_gbestfit = cr_currbestfit




    cr_gbest, cr_gbestfit = get_global_best(REEFpob, REEFfitness)
    testAcc = test_accuracy(cr_gbest, trainX, testX, trainy, testy)
    featCnt = onecnt(cr_gbest)
    print("Test Accuracy: ", testAcc)
    print("#Features: ", featCnt)

    return testAcc, featCnt

datasetlist = ["BreastCancer.csv", "BreastEW.csv", "CongressEW.csv", "Exactly.csv", "Exactly2.csv", "HeartEW.csv", "Ionosphere.csv", "Lymphography.csv", "M-of-n.csv", "PenglungEW.csv", "Sonar.csv", "SpectEW.csv", "Tic-tac-toe.csv", "Vote.csv","Wine.csv", "Zoo.csv","KrVsKpEW.csv",  "WaveformEW.csv"]

for datasetname  in datasetlist:
    accuArr = []
    featArr = []
    for i in range(15):
        testAcc, featCnt = coral_reef(datasetname)
        accuArr.append(testAcc)
        featArr.append(featCnt)
    maxx = max(accuArr)
    currFeat= 20000
    for i in range(np.shape(accuArr)[0]):
        if accuArr[i]==maxx and featArr[i] < currFeat:
            currFeat = featArr[i]
    print(datasetname)
    print(maxx,currFeat)
