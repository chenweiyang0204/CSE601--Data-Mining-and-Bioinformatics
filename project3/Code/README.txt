README


***************************
** K - Nearest Neighbors **
***************************

In the knn.py file, if __name__ == '__main__': function assign value to series variables as you like :
(1) filename  : String Type, the data file path you want to run.
(2) K         : Integer Type , the number of final clusters
(2) diff      : Integer Type , for the nominal type, if the value different under same feature then diff is the distance between them. 

After fill up the upper variables, run the script knn.py and will get Accuracy, Recall, Prediction, F_measure print out.



******************
** Naive Bayes **
******************

In the naivebayes.py file, if __name__ == '__main__': function assign value to series variables as you like :
(1) filename  : String Type, the data file path you want to run.

After fill up the upper variables, run the script naivebayes.py and will get Accuracy, Recall, Prediction, F_measure print out.



********************
** Decision Tree **
********************

Decision Tree:

Inputs : Filename, number of folds

To run the code, you can change the filename variable to the specific data you need. It read text documents and store
data into a list.
The number of folds can changed too, depending on the data size.

Ouput :

The results will show the accuracy, precision, recall and f_measure on the run console.


************************
** Random Forest Tree **
************************

Random Forest Tree:

Inputs : Filename, number of folds, number of tree, number of features.

To run the code, you can change the filename variable to the specific data you need. It read text documents and store
data into a list.
The number of folds can changed too, depending on the data size.
The number of tree is also on the main function. Change base on your testing need.
The number of features is also on the main function. 

Ouput :

The results will show the accuracy, precision, recall and f_measure on the run console.




**********************
** Competition Code**
**********************
In the competition.py file, if __name__ == '__main__': function assign value to series variables as you like :
(1) training_set  	 : Fill up the string type training data path.
(2) testing_set   	 : Fill up the string type testing data path.
(2) training_labels      : Fill up the string type training labels path.

After fill up the upper variables, run the script competition.py , it will write a new file named submission.csv
and in the file, file colunm is test data's id and second column is the predition programming make.

