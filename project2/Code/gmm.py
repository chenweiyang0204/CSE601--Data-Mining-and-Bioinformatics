import numpy as np
from scipy.stats import multivariate_normal as mn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import k_mean as km
"""
    Data Pre-Processing Implementation
"""
def pre_processing(filename):
    contents = open(filename, 'r')
    firstLine = contents.readline().split()
    contents.close()
    num = [i for i in range(2,len(firstLine))]

    gene = np.genfromtxt(filename, usecols=num)
    result = np.genfromtxt(filename, usecols=(1), dtype='str').astype(int)
    id = np.genfromtxt(filename, usecols=(0), dtype='str').astype(int)

    return gene, id, result

"""
    PCA_plot Implement
"""
def PCA_plot(gene, geneGroup, cluster, filename):
    if(len(gene[0])==2):
        print("yes")
        M = gene
    else:
        pca = PCA(n_components=2)
        pca.fit(gene)
        M = pca.transform(gene)

    for idx, item in enumerate(geneGroup):
        if len(item) > 0:
            group = [[M[i][0], M[i][1]] for i in item]

                # for x in group:
            group = np.array(group)
            plt.scatter(group[:, 0], group[:, 1], label=("Group " + str(idx+1)))

    plt.title(cluster +" with data : "+ filename + ' - Plot')
    plt.legend(loc=2)
    plt.show()

"""
    Jaccard Coefficient and Rand Index Calculate implement
"""
def Jaccard_Coefficient_and_Rand_Index(test, result):
    # JC = a / (a + b + c)
    groundTrue, check_matrix = np.zeros((len(result),len(result))), np.zeros((len(result),len(result)))
    geneGroup_arr = np.zeros((len(result)))

    for index, item in enumerate(test):
        for x in item:
            geneGroup_arr[x] = index

    M_00, M_01, M_10, M_11 = 0 ,0 ,0 ,0
    for i, iter1 in enumerate(result):
        for j, iter2 in enumerate(result):
            # groundTrue[i,j] = 1 if iter1 == iter2 else 0
            # check_matrix[i,j] = 1 if geneGroup_arr[i] == geneGroup_arr[j] else 0
            if iter1 == iter2:
                if geneGroup_arr[i] == geneGroup_arr[j]:
                    M_11 += 1
                else:
                    M_10 += 1
            else:
                if geneGroup_arr[i] == geneGroup_arr[j]:
                    M_01 += 1
                else:
                    M_00 += 1

    JC = M_11 / (M_11 + M_10 + M_01)

    RandIndex = (M_11 + M_00) / (M_11 + M_00 + M_10 + M_01)

    return JC , RandIndex

def estep(data,k,u,sigma,pi,smoothing):
    retVal = None
    rik = np.zeros((len(data),k))

    # print(denominator)
    # return
    for j in range(k):
        sigma[j] = sigma[j] + smoothing
        numerator = pi[j] * mn.pdf(x=data,mean=u[j],cov=sigma[j],allow_singular=True)
        denominator = pi[0]*mn.pdf(x=data, mean=u[0], cov=sigma[0], allow_singular=True)
        for c in range(1, k):
            denominator += pi[c]*mn.pdf(x=data, mean=u[c], cov=sigma[c], allow_singular=True)
        rik[:,j] = numerator / (denominator)
    retVal = rik
    return retVal

def mstep(data,rik,k):
    update_u = []
    update_sigma = []
    update_pi = []
    for i in range(k):
        T1 = np.sum(rik[:,i],axis=0)
        n = len(data)
        pi = T1/n
        update_pi.append(pi)
        T2 = np.sum(rik[:,i].reshape(len(rik),1)*data,axis=0)
        u = T2 / (T1+0.01)
        update_u.append(u)
        T3 = (rik[:,i].reshape(len(data),1) * (data - u)).T @ (data - u)
        sigma = T3 / (T1+0.01)
        update_sigma.append(sigma)
    return update_u, update_sigma, update_pi

def max_likliehood(data,rik,u,sigma,pi,smoothing):
    T1 = 0
    for k in range(len(rik[0])):
        T1 += np.sum(pi[k]*mn.pdf(data, mean=u[k], cov=sigma[k],allow_singular=True))
    likelihood = np.log(T1)
    return likelihood

def clustering(data,rik):
    find_cluster = np.argmax(rik, axis=1) + 1
    group = {}
    for c in range(1,k+1):
        group[c] = []
    for i in range(len(find_cluster)):
        group[find_cluster[i]].append(i)
    geneGroup = []
    for j in group.keys():
        geneGroup.append(group[j])

    gene = data
    cluster = "Guassian Mixture Model"
    PCA_plot(gene, geneGroup, cluster, filename)

    JC, RD = Jaccard_Coefficient_and_Rand_Index(geneGroup, result)
    print("Jaccard Coefficient is : " + str(JC))
    print("Rand Index is : " + str(RD))

if __name__ == '__main__':
    filename = "cho.txt"
    threshold = 0.000000001
    iteration = 100
    data, id, result = pre_processing(filename)
    smoothing = 0.000000001 * np.ones(len(data[0]))

    k = 5
    central = np.sort(np.random.choice(id,k,replace=False))
    x, y, u = km.k_mean(k,data,central,iteration)
    ds = {1:0,2:0,3:0,4:0,5:0}
    for i in result:
        ds[i] += 1
    pi = [ds[1]/len(data),ds[2]/len(data),ds[3]/len(data),ds[4]/len(data),ds[5]/len(data)]

    # k = 10
    # central = [1,50,100,200,300,400,500,101,102,103]
    # x, y, u = km.k_mean(k,data,central,100)
    # pi = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

    sigma = np.zeros((k,len(data[0]),len(data[0])))
    for i in range(k):
        for j in range(len(sigma[i])):
            x = np.full(shape = len(data[0]), fill_value=2)
            sigma[i][j] = x
    print(sigma.shape)

    # k = 2
    # u = [[0,0],[1,1]]
    # sigma = [[[1,1],[1,1]],[[2,2],[2,2]]]
    # pi = [0.5,0.5]


    likelihood = [0]
    rik = 0
    for it in range(iteration):
        rik = estep(data,k,u,sigma,pi,smoothing)
        u, sigma, pi = mstep(data,rik,k)
        print("Iteration", it+1, ":", "\n")
        print("new u: ", u ,"\n")
        print("new sigma: ", sigma, "\n")
        print("new pi: ", pi, "\n")
        likelihood.append(max_likliehood(data,rik,u,sigma,pi,smoothing))
        print("Log-likelihood: ", likelihood[-1])
        if (likelihood[-1] - likelihood[-2] < threshold):
            break

    clustering(data,rik)





