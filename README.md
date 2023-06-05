Diamond Tip (Giovanni Astante, Letizia Fabbri, Davide Vandelli)

This is the code for the Introduction To Machine Learning competition. To see Vandelli's code, check FluveV repository on Github. Alternatively, check Fabbri's code searching llfabbri repository on GitHub.

Note: the uploaded scripts are the ones used for the first model appearing in the report, at the paragraph "Resnet18 and Euclidean Distance". Trying the other 
models consisted in modifying accordingly single lines of the code used for this model. Whenever a different pretrained ResNet architecture is used, changes
occur to the configuration file and to the lines 15 and 16 of network.py. Whenever the Cosine similarity is used, in test.py "F.pairwise_distance" has to be
changed with "F.cosine_similarity" and "reverse=True" has to be added to the .sort() method.
Moreover, the images in the query and test directories are samples taken from a dataset we've found online.
