
*********************
**  K-Mean **
*********************

In the k-mean.py file, if __name__ == '__main__': function assign value to series variables as you like :
(1) filename  : String Type, the data file path you want to clusters.
(2) K         : Integer Type , the number of final clusters
(3) iteration : Integer Type, Iteration times
(4) central   : Array Type, Contains initial center points' index. the len of array should equal to k.

After fill up the upper variables, run the script k_mean.py and will get PCA plot, and JC, RI value will print out automatic.




***************************************************************
**  Hierarchical Agglomerative clustering with Min Approach **
***************************************************************

In the hierarchical.py file, if __name__ == '__main__': function assign value to series variables as you like :
(1) filename   : String Type, the data file path you want to clusters.
(2) k	       : Integer Type , the number of final clusters

After fill up two variables, run the script hierarchical.py and will get PCA plot, and JC, RI value will print out automatic.




********************************
**  Density-Based Clustering **
********************************
In the density.py file, if __name__ == '__main__': function assign value to series variables as you like :
(1) filename   : String Type, the data file path you want to clusters.
(2) minPts     : Integer Type, the minPts is using to test core point that if there are at least minPts around a point within epsilon radius
(3) epsilon    : Integer Type, radius of points.

After fill up upper variables, run the script density.py and will get PCA plot, and JC, RI value will print out automatic.




********************************
**  GMM Clustering **
********************************
In the density.py file, if __name__ == '__main__': function assign value to series variables as you like :
(1) filename   : String Type, the data file path you want to clusters.
(2) k	       : Integer Type , the number of final clusters
(3) iteration  : Integer Type, Iteration times
(4) central    : Array Type, Contains initial center points' index. the len of array should equal to k.
(5) sigma      : The space of each clusters distribution.
(6) pi         : The percentage of each distributions.




********************************
**  Spectral Clustering **
********************************
In the density.py file, if __name__ == '__main__': function assign value to series variables as you like :
(1) filename   : String Type, the data file path you want to clusters.
(2) k	       : Integer Type , the number of final clusters
(3) sigma      : The distance that the centroids can reach to each point.
(4) central    : Array Type, Contains initial center points' index. the len of array should equal to k.
(5) iteration  : Integer Type, Iteration times

