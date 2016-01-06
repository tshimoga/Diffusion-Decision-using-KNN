# Diffusion-Decision-using-KNN

We are using anaconda distribution of python for this project because of it’s flexibility to install new
packages and it is a specifically made for machine learning. We used scikit-learn(Pedregosa et al.,
2011), numpy, math, sys, scipy, pickle, cpickle, decimal libraries for this project. Please make sure
these libraries available in your version of python before running the code. Make sure the directory
you place the code has the write access to that directory.

1.Run data pickler.py file to perform FDA and pickle the data into .pickle file.

2.Run generate lamdas.py file to generate lamdas for different classes and it’ll be pickled into .pickle
file.

3.Run the corresponding file for the rule as specified below:-

a. DN criteria deltaN.py
b. DV criteria deltaV.py
c. CDV criteria CDV.py
d. PDN criteria bayesian wth N.py
e. PDV criteria bayesian with U.py
f. Multi-class classification using DN deltaN multiclas.py
g. K Nearest Neighbor(binary) KNN.py
h. K Nearest Neighbor(multi class) KNN multiclass.py
