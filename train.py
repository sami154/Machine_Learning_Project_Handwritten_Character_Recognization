# importing libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from skimage.feature import hog
import torch

#*******************************************************Loading inpput data********************************
train_data_path_1='C:/Users/newuser/OneDrive - University of Florida/Course Works/fall 19/machine learning/assignment/project/empty string/ML project_jyoti/train_data.pkl'
train_data_labs_path_1='C:/Users/newuser/OneDrive - University of Florida/Course Works/fall 19/machine learning/assignment/project/empty string/ML project_jyoti/finalLabelsTrain.npy'
#train_data_path_2='C:/Users/newuser/OneDrive - University of Florida/Course Works/fall 19/machine learning/assignment/project/empty string/ML project_jyoti/known_data.npy' #extra data created by us
#train_data_labs_path_2='C:/Users/newuser/OneDrive - University of Florida/Course Works/fall 19/machine learning/assignment/project/empty string/ML project_jyoti/known_label.npy' #extra data label created by us
model_path='C:/Users/newuser/OneDrive - University of Florida/Course Works/fall 19/machine learning/assignment/project/empty string/ML project_jyoti/extra_model.pth'


# for loading data
def load_pkl(fname):
    with open(fname,'rb') as f:
        return pickle.load(f)

# to save pickel
def save_pkl(fname,obj):
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

#*******************************************************************loading training data***********************************************
def train_func (X,Y):

    # saving the data as array and list
    train_data=list(load_pkl(X))
    train_data_labs=np.load(Y)
#    train_data_1=list(np.load(train_data_path_2,allow_pickle=True))
#    train_data_1_labs=np.load(train_data_labs_path_2)
    
    # if considering extra data, adding those to original data, otherwise just contiuing with original data given by teacher
    train_data=train_data
    train_labs=train_data_labs
#    train_data.extend(train_data_1)
#    train_labs=np.hstack([train_data_labs,train_data_1_labs])
#    
    
    # reading every image and finding which image has largest rown and columns because later we will reshape other images to have the same dimension
    max_first_ax=0
    max_second_ax=0
    for i,j in enumerate(train_data):
        try:
            if max_first_ax<np.array(j).shape[0]:
                max_first_ax=np.array(j).shape[0]
    
            if max_second_ax<np.array(j).shape[1]:
                max_second_ax=np.array(j).shape[1]
        except Exception as e:
            pass
            #print(i,e)
    
    final_train_data=np.zeros(dtype='float32',shape=(len(train_data),max_first_ax,max_second_ax))
    final_train_labs=np.asarray(train_labs).astype('float32')
    
    # padding all images with black spots to make all images of equal size
    for j in range(final_train_data.shape[0]):
        final_train_data[j,0:np.array(train_data[j]).shape[0],0:np.array(train_data[j]).shape[1]]=train_data[j]
    
    
    # desired image dimensions for all images
    img_rows, img_cols = max_first_ax, max_second_ax
    
    # the data, split between train and test sets
    x_train, x_test, y_train, y_test = train_test_split(final_train_data,final_train_labs,test_size=0.2,random_state=35)
    
    # final input data for training model
    features_list = []
    for k in range(len(x_train)):
        df= hog(x_train[k], orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))
        features_list.append(df)
    
    # final validation data for testing the model with a, b class only
    how_many_ab=y_test[y_test < 3]
    x_test_ab=np.zeros(dtype='float32',shape=(len(how_many_ab),max_first_ax,max_second_ax))
    y_test_ab=[]
    c1=0
    for i in range(len(y_test)):
        temp_label=y_test[i]
        #print("temp_label"+str(temp_label))
        if temp_label<3.0:
            x_test_ab[c1]=x_test[i]
            y_test_ab.append(y_test[i])
            c1=c1+1
    y_test_ab=np.array(y_test_ab) 
    
    test_list_ab = []
    for k in range(len(x_test_ab)):
        df1_ab= hog(x_test_ab[k], orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))
        test_list_ab.append(df1_ab)
        
    # final validation data for testing the model with all 8 classes
    test_list= []
    for k in range(len(x_test)):
        df1= hog(x_test[k], orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))
        test_list.append(df1)
            
        
    #****************training the model**************    
    model_score_when8classes=[]
    model_score_when2classes=[]
    k=[]
    for l in range(15):
         temp_k=l+1
         knn_model = neighbors.KNeighborsClassifier(n_neighbors=temp_k)    
         knn_model.fit(features_list,y_train) 
         temp_model_score_8class = knn_model.score(test_list,y_test)
         temp_model_score_2class = knn_model.score(test_list_ab,y_test_ab)
         model_score_when8classes.append(temp_model_score_8class)
         model_score_when2classes.append(temp_model_score_2class)
         k.append(temp_k)
      
    ## checking robustness
    #knn_model_consistency_check = neighbors.KNeighborsClassifier(n_neighbors=7)    
    #knn_model_consistency_check.fit(features_list,y_train) 
    #temp_model_score_8class_consistency_check = knn_model_consistency_check.score(test_list,y_test)
    #temp_model_score_2class_consistency_check  = knn_model_consistency_check.score(test_list_ab,y_test_ab)
    #print("val acc for temp_model_score_8class "+str(temp_model_score_8class_consistency_check))
    #print("val acc for temp_model_score_2class "+str(temp_model_score_2class_consistency_check))
    
    # final model
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=7)    
    knn_model.fit(features_list,y_train)
    return (knn_model)
    
 
    
#main function:
trained_model=train_func (train_data_path_1,train_data_labs_path_1)
torch.save(trained_model,model_path)



   
    
## plot 1: k selection
#import pylab
#fig = plt.figure(figsize = (7,4))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('no. of neighbors (k)', fontsize = 14)
#ax.set_ylabel('validation accuracy (%)', fontsize = 14)
#plt.xticks(fontsize = 14) # work on current fig
#plt.yticks(fontsize = 14) # work on current fig
#plt.plot(k, model_score_when8classes,color='r',lw=2,label='no. of class=8')
#plt.plot(k, model_score_when2classes,color='b',lw=2,label='no. of class=2')
#pylab.legend(loc='upper right',fontsize = 12)
#plt.axvline(x=7,color='black',lw=2, linestyle='--')
##pylab.ylim(0, 170)
#pylab.show()    
#





## plot 2: fetaure sleection
#import pylab
#fig = plt.figure(figsize = (7,4))
#ax = fig.add_subplot(1,1,1) 
#x_stick_desired=['original','normalized','pca','hog']
#x=[1,2,3,4]
#y=[72,80,92,95]
#at_which_index_to_write = [1,2,3,4]
#ax.set_xlabel('different feature selection method', fontsize = 14)
#ax.set_ylabel('validation accuracy (%)', fontsize = 14)
#plt.xticks(at_which_index_to_write, x_stick_desired,fontsize = 14) # work on current fig
#plt.yticks(fontsize = 14) # work on current fig
#plt.plot(x, y,lw=2,marker='o')
##pylab.legend(loc='upper right',fontsize = 12)
##pylab.ylim(0, 170)
#pylab.show()    
#
#
#
## plot 2: classifier selection
#import pylab
#fig = plt.figure(figsize = (7,4))
#ax = fig.add_subplot(1,1,1) 
#x_stick_desired=['svm','cnn','knn']
#x=[1,2,3]
#y=[91,94,95]
#at_which_index_to_write = [1,2,3]
#ax.set_xlabel('different classification method', fontsize = 14)
#ax.set_ylabel('validation accuracy (%)', fontsize = 14)
#plt.xticks(at_which_index_to_write, x_stick_desired,fontsize = 14) # work on current fig
#plt.yticks(fontsize = 14) # work on current fig
#plt.plot(x, y,lw=2,marker='o')
##pylab.legend(loc='upper right',fontsize = 12)
##pylab.ylim(0, 170)
#pylab.show() 
#
#
## checking consistency of validation accurcay in knn
## in order to do these
#import pylab
#fig = plt.figure(figsize = (7,4))
#ax = fig.add_subplot(1,1,1) 
#x_stick_desired=['no. of class = 2','no. of class = 8']
#x=[1,2]
#y1_8class=[0.954,0.940,0.951,0.945,0.955]
#y2_2class=[0.986,0.947,0.965,0.945,0.985]
#at_which_index_to_write = [1,2,3]
#ax.set_xlabel('different classification method', fontsize = 14)
#ax.set_ylabel('validation accuracy (%)', fontsize = 14)
#plt.xticks(at_which_index_to_write, x_stick_desired,fontsize = 14) # work on current fig
#plt.yticks(fontsize = 14) # work on current fig
#plt.plot(x, y,lw=2,marker='o')
##pylab.legend(loc='upper right',fontsize = 12)
##pylab.ylim(0, 170)
#pylab.show() 
#
## new plot
###*******************plot
#data = [[0.954,0.940,0.951,0.945,0.955],[0.986,0.947,0.965,0.949,0.98]]
#x=[1,2]
#x_stick_desired=['no. of class = 8','no. of class = 2']
#at_which_index_to_write =[1,2]
#boxprops = dict(linestyle='-', linewidth=2, color='blue')
#flierprops = dict(marker='o', markerfacecolor='green', markersize=3,linestyle='none')
#medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
#fig = plt.figure(figsize = (6,4))
#ax = fig.add_subplot(1,1,1)
#ax.boxplot(data, boxprops=boxprops,flierprops=flierprops,medianprops=medianprops)
##ax.set_title('different test sets', fontsize=14)
#plt.xticks(at_which_index_to_write, x_stick_desired,fontsize = 14) # work on current fig
#plt.yticks(fontsize = 14) #
#ax.set_xlabel('different test sets', fontsize = 14)
#ax.set_ylabel('validation accuracy (%)', fontsize = 14)
##if u wnat to show marker/ pitns in plot:plt.plot(x, y,marker='o', color='b')
#plt.show()
