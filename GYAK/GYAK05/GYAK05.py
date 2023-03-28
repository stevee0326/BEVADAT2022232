
import numpy as np
import seaborn as sns
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

class KNNClassifier:
    
    @property
    def k_neighbours(self):
        return self.k
    
    def __init__(self, k:int, test_split_ratio:float):
        self.k = k
        self.test_split_ration = test_split_ratio
        
    def train_test_split(self,features:np.ndarray,labels:np.ndarray) -> None:
        
        test_size = int(len(features) * test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        self.x_train, self.y_train = features[:train_size,:],labels[:train_size]
        self.x_test, self.y_test = features[train_size:train_size+test_size,:], labels[train_size:train_size + test_size]
    
    def euclidean(self, element_of_x:np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((self.x_train - element_of_x)**2,axis=1))
    
    def predict(self, x_test:np.ndarray):
        labels_pred = []
        for x_test_element in x_test:
            distances = self.euclidean(self.x_train,x_test_element)
            distances = np.array(sorted(zip(distances, self.y_train)))
            label_pred = mode(distances[:self.k,1],keepdims=False).mode
            labels_pred.append(label_pred)
        self.pred = np.array(labels_pred,dtype=np.int32)
    
    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100
    
    def confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test,self.y_preds)
        sns.heatmap(conf_matrix,annot=True) 
        
    @staticmethod
    def load_csv(csv_path:str) -> Tuple[np.ndarray,np.ndarray]:
        np.random.seed(42)
        dataset = np.genfromtxt(csv_path,delimiter=',')
        np.random.shuffle(dataset,)
        x,y = dataset[:,:4],dataset[:,-1]
        x[np.isnan(x)] = 3.5
        y = np.delete(y,np.where(x < 0.0)[0],axis=0)
        y = np.delete(y,np.where(x > 13.0)[0],axis=0)
        x = np.delete(x,np.where(x < 0.0)[0],axis=0)
        x = np.delete(x,np.where(x > 13.0)[0],axis=0)
        return x,y

