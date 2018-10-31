from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import pandas as pd
import pickle


timit_training_dataset = pd.read_hdf("./Training_Data/mfcc/timit.hdf")
timit_training_dataset.head()

phonemes = timit_training_dataset['labels'].unique().tolist()#find unique values
print(phonemes)
phoneme_wise_list = []

# seperating data by phoneme
for i in range (len(phonemes)):
    phoneme_wise_list.append(timit_training_dataset[timit_training_dataset['labels']==phonemes[i]]) 

timit_mfcc_delta = pd.read_hdf("./Training_Data/mfcc_delta/timit.hdf")
timit_mfcc_delta_delta = pd.read_hdf("./Training_Data/mfcc_delta_delta/timit.hdf")

list_mfcc_delta = []
list_mfcc_delta_delta = []

for i in range (len(phonemes)):
    list_mfcc_delta.append(timit_mfcc_delta[timit_mfcc_delta['labels']==phonemes[i]])

for i in range (len(phonemes)):
    list_mfcc_delta_delta.append(timit_mfcc_delta_delta[timit_mfcc_delta_delta['labels']==phonemes[i]])


# Training 2 mixutre GMM with MFCC features with Energy Coefficients
models=[]

for i in range (len(phoneme_wise_list)):
    features = np.array(phoneme_wise_list[i]["features"].tolist())
    models.append(GMM(n_components=2,covariance_type='diag').fit(features))
    path = "models//2_with_EC//"+phonemes[i]+".pkl"
    pickle.dump(models[i] , open(path, 'wb'))
    


# Training 2 mixutre GMM with MFCC delta features with Energy Coefficients
models=[]

for i in range (len(list_mfcc_delta)):
    features = np.array(list_mfcc_delta[i]["features"].tolist())
    models.append(GMM(n_components=2,covariance_type='diag').fit(features))
    path = "models//2_delta_with_EC//"+phonemes[i]+".pkl"
    pickle.dump(models[i] , open(path, 'wb'))


# Training 2 mixutre GMM with MFCC delta-delta features with Energy Coefficients
models=[]

for i in range (len(list_mfcc_delta_delta)):
    features = np.array(list_mfcc_delta_delta[i]["features"].tolist())
    models.append(GMM(n_components=2,covariance_type='diag').fit(features))
    path = "models//2_delta_delta_with_EC//"+phonemes[i]+".pkl"
    pickle.dump(models[i] , open(path, 'wb'))


# Training 2 mixutre GMM with MFCC delta features without Energy Coefficients
models=[]

for i in range (len(list_mfcc_delta)):
    features = np.array(list_mfcc_delta[i]["features"].tolist())
    temp1 = features[:,1:12]
    temp2 = features[:,14:25]
    features = np.hstack((temp1, temp2))
    models.append(GMM(n_components=2,covariance_type='diag').fit(features))
    path = "models//2_delta//"+phonemes[i]+".pkl"
    pickle.dump(models[i] , open(path, 'wb'))
    


# Training 2 mixutre GMM with MFCC delta-delta features without Energy Coefficients
models=[]

for i in range (len(list_mfcc_delta_delta)):
    features = np.array(list_mfcc_delta_delta[i]["features"].tolist())
    temp1 = features[:,1:12]
    temp2 = features[:,14:25]
    temp3 = features[:,27:38]
    features = np.hstack((temp1, temp2, temp3))
    models.append(GMM(n_components=2,covariance_type='diag').fit(features))
    path = "models//2_delta_delta//"+phonemes[i]+".pkl"
    pickle.dump(models[i] , open(path, 'wb'))


#  Training the GMM with different number of mixtures for MFCC features with no delta features and No energy co-eficients.

for mixtures in [2, 4, 8, 16, 32, 64, 128, 256]:
    models=[]
    for i in range (len(phoneme_wise_list)):
        features = np.array(phoneme_wise_list[i]["features"].tolist())
        features = features[:,1:]
        models.append(GMM(n_components=mixtures,covariance_type='diag').fit(features))
        path = "models//"+str(mixtures)+"//"+phonemes[i]+".pkl"
        pickle.dump(models[i] , open(path, 'wb'))
    models[0].weights_