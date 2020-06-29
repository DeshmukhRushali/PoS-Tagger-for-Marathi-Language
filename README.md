# marathi_pos_rad_3NOV17.pos
It is Part of Speech Tagged dataset of 1500 sentences.


# pos_tagger_mc test.py

PoS Tagger is used to assign a grammatical category to each word of a sentence of Marathi Language. This code is implemented using sequence taggers as well as machine learning methods. Model evaluation is done using the holdout method, in which training is done using 2/3 data, and 1/3 is used for test purpose. Each possible pair of models is compared using the Mcnemar test.

#pos_tagge_kfold.py
This code is implemented using sequence taggers as well as machine learning methods. Model evaluation is done using the k fold cross-validation method.

#pos_t_test.py
A T-test is performed on a pair of taggers to check whether they are the same, or there is a statistical difference between them.

#HMM_PoS Tagger for Marathi_Kfold.py
Hidden Markov Model is implemented using k fold cross-validation for PoS tagging.
