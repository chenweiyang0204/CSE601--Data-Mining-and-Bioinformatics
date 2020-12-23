import numpy as np
import k_mean as km
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def readfile(filename):
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
            if(idx == len(geneGroup)-1):
                plt.scatter(group[:, 0], group[:, 1], label=("Group" + str(idx+1)))
            else:
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

def preprocessing(data,sigma):
    similarity_matrix = np.zeros((len(data), len(data)))
    for gene_1 in data.keys():
        for gene_2 in data.keys():
            dist = np.linalg.norm(data[gene_1]-data[gene_2])
            Wij = np.exp(-((dist**2)/(sigma**2)))
            similarity_matrix[gene_1-1][gene_2-1] = Wij
        # break
    # print(similarity_matrix)
    degree_matrix = np.zeros((len(data), len(data)))
    for i in range(len(degree_matrix)):
        deg = np.sum(similarity_matrix[i])
        degree_matrix[i][i] = deg
    # print(degree_matrix)
    laplacian_matrix = (degree_matrix-similarity_matrix)
    # print(laplacian_matrix[0])
    return laplacian_matrix

def decomposition(L):
    eigen_val, eigen_vect = np.linalg.eig(L)
    sort_eigen_idx = eigen_val.argsort()
    delta = 0
    d = 0
    for i in range(1,len(eigen_val)):
        gap = abs(eigen_val[sort_eigen_idx[i]] - eigen_val[sort_eigen_idx[i-1]])
        if gap > delta:
            d = sort_eigen_idx[i]
            delta = gap
    new_space = sort_eigen_idx[:d]
    return eigen_vect[:,new_space]
    # return eigen_vect

def clustering(data, eigen_vect, k, iteration, central):
    id = []
    gene = []
    for key in data.keys():
        gene.append(data[key])
        id.append(key)
    error,geneGroup,center = km.k_mean(k,eigen_vect,central,iteration)

    return gene,geneGroup

if __name__ == '__main__':
    filename = "cho.txt"
    # filename = "new_dataset_1.txt"
    data, id, result = readfile(filename)
    datakey = {}
    for i in range(len(data)):
        datakey[i] = data[i]
    iteration = 100

    # k = 3
    # sigma = 1
    # central = [3,5,9]

    k = 5
    sigma = 0.1
    central = [1,50,150,200,250]

    L = preprocessing(datakey,sigma)
    eigen_vect = decomposition(L)
    gene, geneGroup = clustering(datakey, eigen_vect, k, iteration, central)

    cluster = "K_Spectral"
    PCA_plot(gene,geneGroup,cluster,filename)

    JC, RD = Jaccard_Coefficient_and_Rand_Index(geneGroup, result)
    print("Jaccard Coefficient is : " + str(JC))
    print("Rand Index is : " + str(RD))