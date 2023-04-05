import pandas as pd
from scipy.stats import mode
from typing import Tuple
from sklearn.metrics import confusion_matrix
import seaborn as sns

class KNNClassifier:
    
    def __init__(self, k:int, test_split_ratio:float):
        self.k = k
        self.test_split_ratio = test_split_ratio

    @property
    def k_neighbors(self):
        return self.k

    def train_test_split(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        self.x_train, self.y_train = features.iloc[:train_size, :], labels.iloc[:train_size]
        self.x_test, self.y_test = features.iloc[train_size:train_size + test_size, :], labels.iloc[train_size:train_size + test_size]
        
    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        labels_pred = []
        for _, x_test_element in x_test.iterrows():
            distances = self.euclidean(x_test_element)
            distances = pd.DataFrame({'distance': distances, 'label': self.y_train}).sort_values(by='distance')
            label_pred = mode(distances.head(self.k)['label']).mode[0]
            labels_pred.append(label_pred)
        self.y_preds = pd.Series(labels_pred)
    
    def euclidean(self, element_of_x: pd.Series) -> pd.Series:
        return ((self.x_train - element_of_x) ** 2).sum(axis=1).pow(0.5)

    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100

    def plot_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_preds)

    def best_k(self):
        k_min = 1 
        k_max = 20
        best_k_value, best_accuracy = 0, 0
        for k in range(k_min, k_max + 1):
            self.k = k
            y_preds = self.predict(self.x_test)
            acc = self.accuracy()
            if acc > best_accuracy:
                best_k_value, best_accuracy = k, acc
        return best_k_value, round(best_accuracy, 2)

    @staticmethod
    def load_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        dataset = pd.read_csv(csv_path)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        return x, y