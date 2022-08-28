# importing libraries
import numpy as np
import pickle
#from PIL import Image
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from skimage.feature import hog
#import math
import torch

# for loading data
def load_pkl(fname):
    with open(fname,'rb') as f:
        return pickle.load(f)

# to save pickel
def save_pkl(fname,obj):
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

#********************************************if you want to predicting labels only*******************************************
model_path='C:/Users/dhwan/OneDrive/Desktop/ML Project_test/Submit_ML/digit_recognition1.pth'
test_data_path='C:/Users/dhwan/OneDrive/Desktop/ML Project_test/Submit_ML/known_data_11.26.19.npy'

        
def testing_predict(unknown_data,file_path):
    final_data=np.zeros(dtype='float32',shape=(len(unknown_data),54,50))
    
    # padding all images with black spots to make all images of equal size
    for j in range(final_data.shape[0]):
        try:
            final_data[j,0:unknown_data[j].shape[0],0:unknown_data[j].shape[1]]=unknown_data[j].astype('float32')
        except:
            pass
    
            
    test_list_unknown= []
    for k in range(len(final_data)):
        df_test_unknowndata= hog(final_data[k], orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))
        test_list_unknown.append(df_test_unknowndata)


    trained_model = torch.load(file_path)
    labels_unknown = trained_model.predict(test_list_unknown)
    labels_vector=np.asarray(labels_unknown)
    return (labels_vector)

# loading data
test_data_new=np.load(test_data_path,allow_pickle=True)
output_labels=testing_predict(test_data_new,model_path)
print(output_labels)



#********************************************if you want to see accuracy scores only*******************************************
#model_path='C:/Users/dhwan/OneDrive/Desktop/ML Project_test/Submit_ML/digit_recognition1.pth'
#test_data_path='C:/Users/dhwan/OneDrive/Desktop/ML Project_test/Submit_ML/known_data_11.26.19.npy'
#test_label_path='C:/Users/dhwan/OneDrive/Desktop/ML Project_test/Submit_ML/known_label_11.26.19.npy'
#
#
#
#def testing_score(unknown_data,unknown_data_labs,file_path):
#    final_data=np.zeros(dtype='float32',shape=(len(unknown_data),54,50))
#    final_labs=np.asarray(unknown_data_labs).astype('float32')
#    
#    # padding all images with black spots to make all images of equal size
#    for j in range(final_data.shape[0]):
#        try:
#            final_data[j,0:unknown_data[j].shape[0],0:unknown_data[j].shape[1]]=unknown_data[j].astype('float32')
#        except:
#            pass
#    
#            
#    test_list_unknown= []
#    for k in range(len(final_data)):
#        df_test_unknowndata= hog(final_data[k], orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))
#        test_list_unknown.append(df_test_unknowndata)
#
#
#    trained_model = torch.load(file_path)
#    score_unknown = trained_model.score(test_list_unknown,final_labs)
#    return (score_unknown)
#
## loading data
#test_data_new=np.load(test_data_path,allow_pickle=True)
#test_label_new=np.load(test_label_path)
#testing_acc=testing_score(test_data_new,test_label_new,model_path)
#print(testing_acc)