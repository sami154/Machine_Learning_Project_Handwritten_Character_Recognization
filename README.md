	PACKAGES Needed to run the code:
	
	1.	Numpy
	2.	Pickle
	3.	Scikit-learn
	4.	Scikit-image
	5.	Scipy
	6.	Torch
	7.	Matplotlib (it is commented in the code, could be used to see graphs in train.py)

	Number of files included:
	
	1.	Train.py
	2.	Test.py
	3.	Model: digit_recognition1.pth
	4.	Extra data used for training (created by us): known_data.npy and known_label.npy
	5.	Data given by teacher for training: train_data.pkl and finalLabelsTrain.npy

	To Run test.py (Note: There are 2 parts of the code)
	
	1.	For getting labels
			Change directories in line number 33 and 34 (33: Model path, 34: test data). 
			Run the code
			Model name: digit_recognition1.pth
	2.	For checking accuracy
			First uncomment lines starting from 66 till end (line number 99).
			Change directories in line number 67, 68 and 69 (67: model path, 68: testing data, 69: testing labels).
			Run the entire code to check the accuracy.

	To Run train.py (Note there are 2 ways the code can be run)
	
	1.	To Run the train.py on Train data provided by teacher
			Change directories in line number 21, 22, and 25 (21: train data, 22: train data label, 25: where you want to save the model).
			Run the code
			Note: the name for the model is different in train.py than the model provided for test.py since we want to avoid overwriting of the model. The output for the train.py will be the model named “extra_model.pth” 
			To verify the plots we created, uncomment the matplotlib library and the line numbers 140 till the end (line number 232)
	2.	To Run the train.py on train data provided by teacher plus additional data created by us.
			Copy the additional data known_data.npy and known_label.npy) into the same folder as the train.py file.
			Uncomment the following line numbers: 23, 24, 44, 45, 50 and 51.
			Change directories in line number 23 and 24 (23: known_data, 24: known_label)
			Run the code
