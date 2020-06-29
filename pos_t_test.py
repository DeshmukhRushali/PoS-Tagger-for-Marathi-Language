import numpy as np
import math
#error_nb=[0.21,0.21,0.21,0.20,0.20]
#error_dt=[0.23,0.23,0.23,0.22,0.21]
#error_nn = [0.19,0.18,0.17,0.17,0.16]
#error_knn = [0.25,0.22,0.23,0.23,0.22]
#error_hmm = [0.21,0.21,0.21,0.22,0.22]
#error_crf = [0.16,0.15,0.16,0.16,0.14]

error_nb=[0.22,0.22,0.20,0.21,0.21]
error_dt=[0.22,0.23,0.22,0.22,0.22]
error_nn = [0.18,0.19,0.17,0.16,0.17]
error_knn = [0.23,0.23,0.22,0.22,0.22]
error_hmm = [0.21,0.21,0.21,0.22,0.22]
error_crf = [0.17,0.17,0.15,0.16,0.15]
error_rf =[0.21,0.21,0.20,0.20,0.20]
print("for k fold . degree of freedom = k-1")
print("For significance level 5% i.e. 0.05 , z= 0.025")
print("t > Z or t < -Z then reject null hypothesis")
print("here z value = 2.776")


M1 = np.mean(error_nb)
M2=np.mean(error_dt)
print("Mean error of NB=",M1,"Mean error of DT=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_nb[i]-error_dt[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nb_dt = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Naive Bayes and Decision tree" ,t_nb_dt)

# NB and NN
M1 = np.mean(error_nb)
M2=np.mean(error_nn)
print("Mean error of NB=",M1,"Mean error of NN=",M2)

sum1=0
for i in range(0,5):
    s1=math.pow((error_nb[i]-error_nn[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nb_nn = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Naive Bayes and Neural network" ,t_nb_nn)


# NB and KNN
M1 = np.mean(error_nb)
M2=np.mean(error_knn)
print("Mean error of NB=",M1,"Mean error of KNN=",M2)

sum1=0
for i in range(0,5):
    s1=math.pow((error_nb[i]-error_knn[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nb_knn = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Naive Bayes and K nearest neighbour" ,t_nb_knn)

# NB and HMM
M1 = np.mean(error_nb)
M2=np.mean(error_hmm)

print("Mean error of NB=",M1,"Mean error of HMM=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_nb[i]-error_hmm[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nb_hmm = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Naive Bayes and hidden markov model" ,t_nb_hmm)

# NB and CRFs
M1 = np.mean(error_nb)
M2=np.mean(error_crf)
print("Mean error of NB=",M1,"Mean error of CRF=",M2)

sum1=0
for i in range(0,5):
    s1=math.pow((error_nb[i]-error_crf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nb_crf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Naive Bayes and Conditional Random fields" ,t_nb_crf)



# NB and RFs
M1 = np.mean(error_nb)
M2=np.mean(error_rf)

print("Mean error of NB=",M1,"Mean error of RF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_nb[i]-error_rf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nb_rf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Naive Bayes and Random forest classifier" ,t_nb_rf)






# DT and NN
M1 = np.mean(error_dt)
M2=np.mean(error_nn)

print("Mean error of DT=",M1,"Mean error of NN=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_dt[i]-error_nn[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_dt_nn = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Decision tree and Neural Network" ,t_dt_nn)


# DT and KNN
M1 = np.mean(error_dt)
M2=np.mean(error_knn)

print("Mean error of DT=",M1,"Mean error of KNN=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_dt[i]-error_knn[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_dt_knn = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Decision tree and K nearest neighbour" ,t_dt_knn)


# DT and HMM
M1 = np.mean(error_dt)
M2=np.mean(error_hmm)

print("Mean error of DT=",M1,"Mean error of HMM=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_dt[i]-error_hmm[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_dt_hmm = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Decision tree and hidden markov model" ,t_dt_hmm)

# DT and CRF
M1 = np.mean(error_dt)
M2=np.mean(error_crf)

print("Mean error of DT=",M1,"Mean error of CRF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_dt[i]-error_crf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_dt_crf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Decision tree and conditional random model" ,t_dt_crf)

# DT and RF
M1 = np.mean(error_dt)
M2=np.mean(error_rf)

print("Mean error of DT=",M1,"Mean error of RF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_dt[i]-error_rf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_dt_rf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between Decision tree and conditional random model" ,t_dt_rf)



# NN and KNN
M1 = np.mean(error_nn)
M2=np.mean(error_knn)

print("Mean error of NN=",M1,"Mean error of KNN=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_nn[i]-error_knn[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nn_knn = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between neural network and K nearest neighbour" ,t_nn_knn)

# NN and HMM
M1 = np.mean(error_nn)
M2=np.mean(error_hmm)

print("Mean error of NN=",M1,"Mean error of HMM=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_nn[i]-error_hmm[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nn_hmm = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between neural network and hidden markov model" ,t_nn_hmm)


# NN and CRF
M1 = np.mean(error_nn)
M2=np.mean(error_crf)

print("Mean error of NN=",M1,"Mean error of CRF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_nn[i]-error_crf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nn_crf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between neural network and conditional random field" ,t_nn_crf)

# NN and RF
M1 = np.mean(error_nn)
M2=np.mean(error_rf)

print("Mean error of NN=",M1,"Mean error of RF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_nn[i]-error_rf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_nn_rf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between neural network and Random forest classifier",t_nn_rf)






# KNN and HMM
M1 = np.mean(error_knn)
M2=np.mean(error_hmm)

print("Mean error of KNN=",M1,"Mean error of HMM=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_knn[i]-error_hmm[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_knn_hmm = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between k nearest neighbour and hidden markov model" ,t_knn_hmm)


# KNN and CRF
M1 = np.mean(error_knn)
M2=np.mean(error_crf)

print("Mean error of KNN=",M1,"Mean error of CRF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_knn[i]-error_crf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_knn_crf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between k nearest neighbour and Conditional Random fields" ,t_knn_crf)

# KNN and RF
M1 = np.mean(error_knn)
M2=np.mean(error_rf)

print("Mean error of KNN=",M1,"Mean error of RF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_knn[i]-error_rf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_knn_rf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between k nearest neighbour and Random forest classifier" ,t_knn_rf)



# HMM and CRF
M1 = np.mean(error_hmm)
M2=np.mean(error_crf)

print("Mean error of HMM=",M1,"Mean error of CRF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_hmm[i]-error_crf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_hmm_crf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between hidden markov model and Conditional Random fields" ,t_hmm_crf)


# HMM and RF
M1 = np.mean(error_hmm)
M2=np.mean(error_rf)

print("Mean error of HMM=",M1,"Mean error of RF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_hmm[i]-error_rf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_hmm_rf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between hidden markov model and Random forest classifier" ,t_hmm_rf)



# CRF and RF
M1 = np.mean(error_crf)
M2=np.mean(error_rf)

print("Mean error of CRF=",M1,"Mean error of RF=",M2)
sum1=0
for i in range(0,5):
    s1=math.pow((error_crf[i]-error_rf[i]-(M1-M2)),2)
    sum1 = sum1+ s1 
               
varM1_M2=sum1/5

t_crf_rf = (M1-M2)/math.sqrt(varM1_M2/5)

print("t test value between conditional random field and Random forest classifier" ,t_crf_rf)






























