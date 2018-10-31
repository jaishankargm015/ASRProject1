from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import pandas as pd
import pickle


with open("phoneme_name.txt", "r") as file:
    phonemes = file.readlines()
phonemes = [ele.strip().split(",") for ele in phonemes]
phonemes = phonemes[0][:-1]

int_to_label=dict((i,c) for i,c in enumerate(phonemes))
label_to_int=dict((c,i) for i,c in enumerate(phonemes))

timit_testing_dataset = pd.read_hdf("./Testing_Data/mfcc/timit.hdf")
timit_testing_dataset_mfcc_delta = pd.read_hdf("./Testing_Data/mfcc_delta/timit.hdf")
timit_testing_dataset_mfcc_delta_delta = pd.read_hdf("./Testing_Data/mfcc_delta_delta/timit.hdf")


testing_data = np.array(timit_testing_dataset["features"].tolist())
testing_data_mfcc_delta = np.array(timit_testing_dataset_mfcc_delta["features"].tolist())
testing_data_mfcc_delta_delta = np.array(timit_testing_dataset_mfcc_delta_delta["features"].tolist())

test_labels = np.array(timit_testing_dataset["labels"].tolist())
test_labels = test_labels.reshape(test_labels.size, 1)
total_number_of_test_samples = testing_data.shape[0]

#testing for mfcc delta and mfcc delta-delta with and without Energy coefficients.

#testing for mfcc delta
gmm = []
for i in range(len(phonemes)):

    path = "models//2_delta//"+phonemes[i]+".pkl"
    with open(path, 'rb') as f:
        gmm.append(pickle.load(f))

matched=0
for i in range (total_number_of_test_samples):
    temp1 = testing_data_mfcc_delta[i,1:12]
    temp2 = testing_data_mfcc_delta[i,14:25]
    temp = np.hstack((temp1, temp2))
    temp = temp.reshape(1,22)
    curr_label = label_to_int[test_labels[i][0]]
    log_likelihood=[]
    for j in range (len(gmm)):
        log_likelihood.append(gmm[j].score(temp))
    ans_label=log_likelihood.index(max(log_likelihood))
    if ans_label==curr_label:
        matched=matched+1
print("Number of Mixtures in each GMM:",2)
print("Number of successful matches: ",matched)
print("Number of mismatches: ",total_number_of_test_samples-matched)
accuracy=(matched/total_number_of_test_samples)*100
print("Accuracy is: ",accuracy)

#testing for mfcc delta delta
gmm = []
for i in range(len(phonemes)):

    path = "models//2_delta_delta//"+phonemes[i]+".pkl"
    with open(path, 'rb') as f:
        gmm.append(pickle.load(f))

matched=0
for i in range (total_number_of_test_samples):
    temp1 = testing_data_mfcc_delta_delta[i,1:12]
    temp2 = testing_data_mfcc_delta_delta[i,14:25]
    temp3 = testing_data_mfcc_delta_delta[i,27:38]
    temp = np.hstack((temp1, temp2, temp3))
    temp = temp.reshape(1,33)
    curr_label = label_to_int[test_labels[i][0]]
    log_likelihood=[]
    for j in range (len(gmm)):
        log_likelihood.append(gmm[j].score(temp))
    ans_label=log_likelihood.index(max(log_likelihood))
    if ans_label==curr_label:
        matched=matched+1
print("Number of Mixtures in each GMM:",2)
print("Number of successful matches: ",matched)
print("Number of mismatches: ",total_number_of_test_samples-matched)
accuracy=(matched/total_number_of_test_samples)*100
print("Accuracy is: ",accuracy)

#testing for mfcc delta with Energy coeffs
gmm = []
for i in range(len(phonemes)):

    path = "models//2_delta_with_EC//"+phonemes[i]+".pkl"
    with open(path, 'rb') as f:
        gmm.append(pickle.load(f))

matched=0
for i in range (total_number_of_test_samples):
    temp = testing_data_mfcc_delta[i,]
    temp = temp.reshape(1,26)
    curr_label = label_to_int[test_labels[i][0]]
    log_likelihood=[]
    for j in range (len(gmm)):
        log_likelihood.append(gmm[j].score(temp))
    ans_label=log_likelihood.index(max(log_likelihood))
    if ans_label==curr_label:
        matched=matched+1
print("Number of Mixtures in each GMM:",2)
print("Number of successful matches: ",matched)
print("Number of mismatches: ",total_number_of_test_samples-matched)
accuracy=(matched/total_number_of_test_samples)*100
print("Accuracy is: ",accuracy)

#testing for mfcc delta delta with Energy coeffs.
gmm = []
for i in range(len(phonemes)):

    path = "models//2_delta_delta_with_EC//"+phonemes[i]+".pkl"
    with open(path, 'rb') as f:
        gmm.append(pickle.load(f))

matched=0
for i in range (total_number_of_test_samples):
    temp = testing_data_mfcc_delta_delta[i,]
    temp = temp.reshape(1,39)
    curr_label = label_to_int[test_labels[i][0]]
    log_likelihood=[]
    for j in range (len(gmm)):
        log_likelihood.append(gmm[j].score(temp))
    ans_label=log_likelihood.index(max(log_likelihood))
    if ans_label==curr_label:
        matched=matched+1
print("Number of Mixtures in each GMM:",2)
print("Number of successful matches: ",matched)
print("Number of mismatches: ",total_number_of_test_samples-matched)
accuracy=(matched/total_number_of_test_samples)*100
print("Accuracy is: ",accuracy)

# Testing Models with different number of Mixture components

for mixtures in [2, 4, 8, 16, 32, 64, 128, 256]:
    gmm = []
    for i in range(len(phonemes)):

        path = "models//"+str(mixtures)+"//"+phonemes[i]+".pkl"
        with open(path, 'rb') as f:
            gmm.append(pickle.load(f))

    matched=0
    for i in range (total_number_of_test_samples):
        temp = testing_data[i,1:]
        temp = temp.reshape(1,12)
        curr_label = label_to_int[test_labels[i][0]]
        log_likelihood=[]
        for j in range (len(gmm)):
            log_likelihood.append(gmm[j].score(temp))
        ans_label=log_likelihood.index(max(log_likelihood))
        if ans_label==curr_label:
            matched=matched+1
    print("Number of Mixtures in each GMM:",mixtures)
    print("Number of successful matches: ",matched)
    print("Number of mismatches: ",total_number_of_test_samples-matched)
    accuracy=(matched/total_number_of_test_samples)*100
    print("Accuracy is: ",accuracy) 