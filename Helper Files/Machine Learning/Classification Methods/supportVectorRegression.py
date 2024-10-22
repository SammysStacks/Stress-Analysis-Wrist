"""
"""

#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.animation as manimation
import joblib
import sys
import os
from sklearn.model_selection import train_test_split
    

class supportVectorRegression:
    def __init__(self, modelPath, modelType = "rbf", polynomialDegree = 3):
        self.polynomialDegree = polynomialDegree
        
        # Plotting Styles
        self.stepSize = 0.01 # step size in the mesh
        self.cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'red']) # Colormap
        self.cmap_bold = ['darkorange', 'c', 'darkblue', 'darkred'] # Colormap
        
        # Initialize Model
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel(modelType)
    
    def saveModel(self, modelPath = "./SVR.sav"):
        joblib.dump(self.model, 'scoreregression.pkl')    
    
    def loadModel(self, modelPath):
        with open(modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        print("SVR Model Loaded")
            
    
    def createModel(self, modelType = "rbf"):
        # C: penatly term. Low C means that we ignore outliers. High C means that we fit perfectly.
        # epsilon: the area around the hyperplane where we will ignore error. Large epsilon will 
        if modelType == "linear":
            self.model = SVR(kernel="linear", C=.1, epsilon=0.193*2)
        elif modelType == "rbf":
            self.model = SVR(kernel="rbf", C=1, gamma='scale', epsilon=0.01)
        elif modelType == "poly":
            self.model = SVR(kernel="poly", C=1, gamma="scale", degree=self.polynomialDegree, epsilon=0.01, coef0=1)
        elif modelType == "sigmoid":
            self.model = SVR(kernel="sigmoid", C=1, gamma='scale', epsilon=0.01, coef0=1)
        elif modelType == "precomputed":
            self.model = SVR(kernel="precomputed", C=1, gamma='scale', epsilon=0.01)
        else:
            print("No SVR Model Matches the Requested Type")
            sys.exit()
        #print("SVR Model Created")
        
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels):  
        # Train the Model
        self.model.fit(Training_Data, Training_Labels)
        modelScore = self.scoreModel(Testing_Data, Testing_Labels)
        return modelScore
    
    def scoreModel(self, signalData, signalLabels):
        return self.model.score(signalData, signalLabels)
    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        return self.model.predict(New_Data)
    
    def plotModel(self, signalData, signalLabels):
        Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.2, shuffle= True, stratify=signalLabels)

        linear = SVR.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        rbf = SVR.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        poly = SVR.SVC(kernel='poly', degree = self.polynomialDegree, C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        sig = SVR.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
    
        #to better understand it, just play with the value, change it and print it
        x_min, x_max = Training_Data[:, 0].min(), Training_Data[:, 0].max()
        y_min, y_max = Training_Data[:, 1].min(), Training_Data[:, 1].max()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, self.stepSize),np.arange(y_min, y_max, self.stepSize))# create the title that will be shown on the plot
        titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']
        
        """
        dimensions = []
        for dimension in range(np.shape(X_train)[1]):
            x_min, x_max = X[:, dimension].min() - 1, X[:, dimension].max() + 1
            dimensions.append(np.arange(x_min, x_max, h))
        xx, yy = np.meshgrid(*dimensions) # create the title that will be shown on the plot
        titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']
        """
        
        for i, clf in enumerate((linear, rbf, poly, sig)):
            FFMpegWriter = manimation.writers['ffmpeg']
            metadata = dict(title="", artist='Matplotlib', comment='Movie support!')
            writer = FFMpegWriter(fps=3, metadata=metadata)
            
            setPointX4 = 0.002;
            errorPoint = 0.003;
            dataWithinChannel4 = Training_Data[abs(Training_Data[:,3] - setPointX4) <= errorPoint]
            
            channel3Vals = np.arange(0.0, dataWithinChannel4[:,2].max(), 0.01)
            
            #defines how many plots: 2 rows, 2columns=> leading to 4 plots
            fig = plt.figure()
            plt.rcParams['figure.dpi'] = 300
            #space between plots
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
            with writer.saving(fig, "./Machine Learning/ML Videos/SVR_" + clf.kernel + ".mp4", 300):
                for setPointX3 in channel3Vals:
            
                    x3 = np.ones(np.shape(xx.ravel())[0])*setPointX3
                    x4 = np.ones(np.shape(xx.ravel())[0])*setPointX4
                    Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), x3, x4])# Put the result into a color plot
                    
                    Z = Z.reshape(xx.shape)
                    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('cubehelix', 6), alpha=0.7, vmin=0, vmax=5)
                    
                    xPoints = []; yPoints = []; yLabelPoints = []
                    for j, point in enumerate(Training_Data):
                        if abs(point[2] - setPointX3) <= errorPoint and abs(point[3] - setPointX4) <= errorPoint:
                            xPoints.append(point[0])
                            yPoints.append(point[1])
                            yLabelPoints.append(Training_Labels[j])
                    
                    plt.scatter(xPoints, yPoints, c=yLabelPoints, cmap=plt.cm.get_cmap('cubehelix', 6), edgecolors='grey', s=50, vmin=0, vmax=5)
                    
                    
                    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')
                    plt.title(titles[i]+": Channel3 = " + str(round(setPointX3,3)) + "; Channel4 = " + str(setPointX4) + "; Error = " + str(errorPoint))
                    plt.xlabel('Channel 1')
                    plt.ylabel('Channel 2')
                    plt.xlim(xx.min(), xx.max())
                    plt.ylim(yy.min(), yy.max())
                    plt.xticks(())
                    plt.yticks(())
                    #plt.title(titles[i])
                    
                    cb = plt.colorbar(ticks=range(6), label='digit value')
                    plt.clim(-0.5, 5.5)
                
                    # Write to Video
                    writer.grab_frame()
                    plt.cla()
                    cb.remove()
        
        
        
        # retrieve the accuracy and print it for all 4 kernel functions
        accuracy_lin = linear.score(Testing_Data, Testing_Labels)
        accuracy_poly = poly.score(Testing_Data, Testing_Labels)
        accuracy_rbf = rbf.score(Testing_Data, Testing_Labels)
        accuracy_sig = sig.score(Testing_Data, Testing_Labels)
        
        print("Accuracy Linear Kernel:", accuracy_lin)
        print("Accuracy Polynomial Kernel:", accuracy_poly)
        print("Accuracy Radial Basis Kernel:", accuracy_rbf)
        print("Accuracy Sigmoid Kernel:", accuracy_sig) 
        
        linear_pred = linear.predict(Testing_Data)
        poly_pred = poly.predict(Testing_Data)
        rbf_pred = rbf.predict(Testing_Data)
        sig_pred = sig.predict(Testing_Data) 
        
        # creating a confusion matrix
        cm_lin = confusion_matrix(Testing_Labels, linear_pred)
        cm_poly = confusion_matrix(Testing_Labels, poly_pred)
        cm_rbf = confusion_matrix(Testing_Labels, rbf_pred)
        cm_sig = confusion_matrix(Testing_Labels, sig_pred)
        
        print(cm_lin)
        print(cm_poly)
        print(cm_rbf)
        print(cm_sig)





