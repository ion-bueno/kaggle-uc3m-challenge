
import numpy as np
import pickle as pkl

# Other modules
from src.utils import *

# Processing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA

# Models
from sklearn.pipeline import Pipeline
#from concurrent.futures import process
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import NearestCentroid

# Keras
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# Fine tuning
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Metrics
from sklearn.metrics import f1_score





class BaselineModel():
    def __init__(self, processing):
        self.model = RandomForestClassifier()
        self.X = processing.X
        self.y = processing.y
        self.X_train = processing.X_train
        self.X_test = processing.X_test
        self.y_train = processing.y_train
        self.y_test = processing.y_test

        self.scaler = processing.scaler
        self.decomposition = processing.decomposition

    def select_model(self):
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
        # number of features at every split
        max_features = ['auto', 'sqrt']

        # max depth
        max_depth = [int(x) for x in np.linspace(5, 135, num = 3)]
        # create random grid
        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth
        }

        rfc_random = RandomizedSearchCV(estimator = self.model, param_distributions = random_grid, n_iter = 30, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring='f1')
        # Fit the model
        rfc_random.fit(self.X_train, self.y_train)

        print(f'Params: {rfc_random.best_params_}, score {rfc_random.best_score_}')

        self.model = rfc_random.best_estimator_

    def predict_(self, test):
        self.model.fit(self.X, self.y)

        test = test.iloc[:, 1:]

        if self.scaler != None:
            test = self.scaler.transform(test)

        if self.decomposition != None:
            test = self.decomposition.transform(test)

        y_pred = self.model.predict(test)
        return y_pred




class Processing():
    def __init__(self, steps, params):
        self.steps = steps
        self.params = params

    def train():
        pass


class Classifier():
    def __init__(self, model, params):
        self.model = model
        self.params = params


class BasicModel():

    def __init__(self, X, y, processing, clf, n_iter=50):
        self.X = X
        self.y = y
        self.processing = processing
        self.clf = clf
        self.hyper_params = {}
        pipe = []

        if processing is not None:
            self.hyper_params.update(processing.params)
            pipe = pipe + self.processing.steps

        self.hyper_params.update(self.clf.params)
        self.search_params = {
            'n_iter': n_iter,
            'cv': 3,
            'verbose': 0
        }
        self.search = BayesSearchCV
        
        pipe.append(self.clf.model)
        self.pipe = Pipeline(steps=pipe)
        self.model = None

    def train(self):
        clf = self.search(
            self.pipe, 
            self.hyper_params, 
            n_iter = self.search_params['n_iter'], 
            cv = self.search_params['cv'], 
            verbose=self.search_params['verbose'], 
            random_state=seed, 
            n_jobs = -1, 
            scoring='f1'
        )
        clf.fit(self.X, self.y)
        print(f'Params:\n{clf.best_params_}')
        print(f'\nF1-score: {clf.best_score_}')
        self.model = clf.best_estimator_
        return self.model




################
# KNN
################

class KNNPCA(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
            #('pca',     PCA(random_state=seed))
        ]
        proc_params = {
            #'pca__n_components': Real(0.7, 0.9999)
        }
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'knn', KNeighborsClassifier()
        )
        model_params = {
            'knn__n_neighbors':   Integer(1, 21)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf, 20)


class NC(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler())
        ]
        proc_params = {}
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'nc', NearestCentroid()
        )
        model_params = {
            'nc__metric':   Categorical(['manhattan'])
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf, 1)


######################
# LogisticRegression
######################


class checkLR(BasicModel):
    def __init__(self, X, y):


        # Classifier
        model = (
            'lr', LogisticRegression(random_state=seed)
        )
        model_params = {
            'lr__max_iter':     Integer(100, 1e4),
            'lr__penalty':      Categorical(['l1', 'l2', 'elasticnet']),
            'lr__C':            Real(1e-1, 10),
            'lr__solver':       Categorical(['saga']),
            'lr__l1_ratio':     Real(0, 1)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        self.search_params['n_iter'] = 2
        super().__init__(X, y, None, clf)


class LRPCA(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
            ('pca',     PCA(random_state=seed))
        ]
        proc_params = {
            'pca__n_components': Real(0.7, 0.95)
        }
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'lr', LogisticRegression(random_state=seed)
        )
        model_params = {
            'lr__max_iter':     Integer(100, 1e4),
            'lr__penalty':      Categorical(['l1', 'l2', 'elasticnet']),
            'lr__C':            Real(1e-1, 10),
            'lr__solver':       Categorical(['saga']),
            'lr__l1_ratio':     Real(0, 1)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        self.search_params['n_iter'] = 2
        super().__init__(X, y, proc, clf)



###################
# DecisionTree
###################

class votingDT(BasicModel):
    def __init__(self , X, y):

        self.pca = PCA(random_state=seed, n_components=0.9)
        self.pca.fit(X)
        self.models = {'svm': None, 'et': None, 'mlp': None}
        self.models['svm'] = pkl.load(open(f'models/model32.pkl', 'rb'))
        self.models['et'] = pkl.load(open(f'models/model54.pkl', 'rb'))
        self.models['mlp'] = tf.keras.models.load_model('models/model45', custom_objects={'f1_metric': f1_metric})

        X_final = self.get_features(X)

        # Classifier
        model = (
            'dt', DecisionTreeClassifier(random_state=seed)
        )
        model_params = {
            'dt__criterion': Categorical(['gini', 'entropy'])
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X_final, y, None, clf, n_iter=2)


    def get_features(self, X):
        X_final = []
        for key, model in self.models.items():
            if 'mlp' in key:
                probs = model.predict(self.pca.transform(X)).reshape(-1,1)
            else:
                probs = model.predict_proba(X)[:,1].reshape(-1,1)
            if len(X_final):
                X_final = np.hstack((X_final, probs))
            else:
                X_final = probs
        return X_final


    def predict(self, X_comp):

        X_final = self.get_features(X_comp)
        return self.model.predict(X_final)



class checkDT(BasicModel):
    def __init__(self , X, y):
        
        random = np.random.normal(0, 0.5, X.shape[0]).reshape(-1,1)
        X = np.hstack((X, random))

        # Classifier
        model = (
            'dt', DecisionTreeClassifier(random_state=seed)
        )
        model_params = {
            'dt__criterion': Categorical(['gini', 'entropy'])
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, None, clf, n_iter=2)
        


################
# RandomForest
################


class checkRF(BasicModel):
    def __init__(self, X, y):

        random = np.random.normal(0, 0.5, X.shape[0]).reshape(-1,1)
        X = np.hstack((X, random))

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
        ]
        proc_params = {}
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'rf', RandomForestClassifier(random_state=seed)
        )
        model_params = {
            'rf__n_estimators':   Integer(100, 1000),
            'rf__criterion':      Categorical(['gini', 'entropy']),
            'rf__max_features':   Categorical(['sqrt', 'log2', None]),
            'rf__max_depth':      Integer(5,135),
            'rf__min_samples_split': Integer(2,10)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)


class RF(BasicModel):
    def __init__(self, X, y):

        # Classifier
        model = (
            'rf', RandomForestClassifier(random_state=seed)
        )
        model_params = {
            'rf__n_estimators':   Integer(100, 1000),
            'rf__criterion':      Categorical(['gini', 'entropy']),
            'rf__max_features':   Categorical(['sqrt', 'log2', None]),
            'rf__max_depth':      Integer(5,135),
            'rf__min_samples_split': Integer(2,10)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, None, clf)


class RFPCA(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
            ('pca',     PCA(random_state=seed))
        ]
        proc_params = {
            'pca__n_components': Real(0.6, 0.95)
        }
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'rf', RandomForestClassifier(random_state=seed)
        )
        model_params = {
            'rf__n_estimators':   Integer(100, 1000),
            'rf__criterion':      Categorical(['gini', 'entropy']),
            'rf__max_features':   Categorical(['sqrt', 'log2', None]),
            'rf__max_depth':      Integer(5,135),
            'rf__min_samples_split': Integer(2,10)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)



################
# ExtraTrees
################

class checkET(BasicModel):
    def __init__(self, X, y):

        random = np.random.normal(0, 0.5, X.shape[0]).reshape(-1,1)
        X = np.hstack((X, random))

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler())
        ]
        proc_params = {
        }
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'et', ExtraTreesClassifier(random_state=seed)
        )
        model_params = {
            'et__n_estimators':   Integer(100, 1000),
            'et__criterion':      Categorical(['gini', 'entropy']),
            'et__max_features':   Categorical(['sqrt', 'log2', None]),
            'et__max_depth':      Integer(5,135),
            'et__min_samples_split': Integer(2,10)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)


class ET(BasicModel):
    def __init__(self, X, y):

        # Classifier
        model = (
            'et', ExtraTreesClassifier(random_state=seed)
        )
        model_params = {
            'et__n_estimators':   Integer(100, 1000),
            'et__criterion':      Categorical(['gini', 'entropy']),
            'et__max_features':   Categorical(['sqrt', 'log2', None]),
            'et__max_depth':      Integer(5,135),
            'et__min_samples_split': Integer(2,10)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, None, clf)


class ETPCA(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
            ('pca',     PCA(random_state=seed))
        ]
        proc_params = {
            'pca__n_components': Real(0.6, 0.95)
        }
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'et', ExtraTreesClassifier(random_state=seed)
        )
        model_params = {
            'et__n_estimators':   Integer(100, 1000),
            'et__criterion':      Categorical(['gini', 'entropy']),
            'et__max_features':   Categorical(['sqrt', 'log2', None]),
            'et__max_depth':      Integer(5,135),
            'et__min_samples_split': Integer(2,10)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)



###################
# Boosting
###################


class HGBPCA(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
            ('pca',     PCA(random_state=seed))
        ]
        proc_params = {
            'pca__n_components': Real(0.7, 0.95)
        }
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'hgb', HistGradientBoostingClassifier(random_state=seed)
        )
        model_params = {
            'hgb__learning_rate': Real(0.001, 0.9),           
            'hgb__max_iter': Integer(400, 800),   
            'hgb__max_leaf_nodes': Integer(5, 100)    
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)



################
# XGBoost
################


class XGBPCA(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
            ('pca',     PCA(random_state=seed))
        ]
        proc_params = {
            'pca__n_components': Real(0.5, 1)
        }
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'xgb', xgb.XGBClassifier(random_state=seed, eval_metric=f1_score, use_label_encoder=False)
        )
        model_params = {
            'xgb__n_estimators':    Integer(200,1300),
            'xgb__eta':             Real(0.01, 0.3),
            'xgb__subsample':       Real(0.3, 0.8),
            'xgb__max_depth':       Integer(5,50)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)


################
# SVM
################


class SVM(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler())
        ]
        proc_params = {}
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'svm', SVC(random_state=seed, probability=True)
        )
        model_params = {
            'svm__kernel':  Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
            'svm__gamma':   Real(1e-5, 1e-1),
            'svm__C':       Real(1e-3, 10),
            'svm__degree':  Integer(1, 5),
            'svm__decision_function_shape': Categorical(['ovo', 'ovr'])
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)



class SVMPCA(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
            ('pca',     PCA(random_state=seed))
        ]
        proc_params = {
            'pca__n_components': Real(0.5, 0.95)
        }
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'svm', SVC(random_state=seed, probability=True)
        )
        model_params = {
            'svm__kernel':  Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
            'svm__gamma':   Real(1e-5, 1e-1),
            'svm__C':       Real(1e-3, 10),
            'svm__degree':  Integer(1, 5),
            'svm__decision_function_shape': Categorical(['ovo', 'ovr'])
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)


class SVMLDA(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
            ('lda',     LinearDiscriminantAnalysis())
        ]
        proc_params = {}
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'svm', SVC(random_state=seed, probability=True)
        )
        model_params = {
            'svm__kernel':  Categorical(['rbf', 'linear']),
            'svm__gamma':   Real(1e-3, 1e-1),
            'svm__C':       Real(10, 100)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)


class SVMKPCA(BasicModel):
    def __init__(self, X, y):

        # Processing
        proc_steps = [
            ('scaler',  StandardScaler()),
            ('kpca',    KernelPCA(random_state=seed))
        ]
        proc_params = {
            'kpca__n_components': Integer(5,200),
            'kpca__kernel':       Categorical(['rbf', 'linear']), 
            'kpca__gamma':        Real(1e-6, 1e-1)
        }
        proc = Processing(proc_steps, proc_params)

        # Classifier
        model = (
            'svm', SVC(random_state=seed, probability=True)
        )
        model_params = {
            'svm__kernel':  Categorical(['rbf', 'linear']),
            'svm__gamma':   Real(1e-3, 1e-1),
            'svm__C':       Real(1, 100)
        }
        clf = Classifier(model, model_params)

        # Model initialization
        super().__init__(X, y, proc, clf)




###################
# MLP
###################


class TFModel():
    def __init__(self, X, y, layers, params, dropout=0, norm=False):

        self.X = X
        self.y = y
        self.params = params
        #tf.random.set_seed(seed)

        self.model = Sequential()
        self.model.add(Dense(layers[0], activation='relu', kernel_initializer='he_normal', input_shape=(self.X.shape[1],)))
        for units in layers[1:]:
            if norm:
                self.model.add(BatchNormalization())
            self.model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))
            if dropout:
                self.model.add(Dropout(dropout))
        self.model.add(Dense(1, activation='sigmoid'))

        self.callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_metric', 
            patience=self.params['patience'], 
            mode='max',
            restore_best_weights=True
        )

    def train(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_metric])
        self.model.fit(self.X, self.y, 
            epochs=self.params['epochs'], 
            batch_size=self.params['batch_size'], 
            verbose=1, 
            validation_split=self.params['val_split'], 
            callbacks=[self.callback]
        )
        probs = self.model.predict(self.X)
        y_pred = predict_threshold(probs, self.params['pred_th'])
        f1 = f1_score(self.y, y_pred)
        print(f'\nF1-score: {f1}')
        return self.model



class BaselineMLP(TFModel):
    def __init__(self , X, y):

        layers = [8,10,10]
        params = {
            'patience': 10,
            'epochs': 60,
            'batch_size': 16,
            'val_split': 0.1,
            'pred_th': 0.5
        }
        # Model initialization
        super().__init__(X, y, layers, params)

    def predict(self, X_comp, threshold):
        probs = self.model.predict(X_comp)
        y_pred = predict_threshold(probs, threshold)
        return y_pred



class MLPPCA(TFModel):
    def __init__(self , X, y):
        
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        self.pca = PCA(random_state=seed, n_components=0.9)
        self.pca.fit(X)

        X = self.scaler.transform(X)
        X = self.pca.transform(X)

        layers = [4,6,4]
        params = {
            'patience': 20,
            'epochs': 300,
            'batch_size': 64,
            'val_split': 0.1,
            'pred_th': 0.5
        }
        # Model initialization
        super().__init__(X, y, layers, params, 0, norm=True)

    def predict(self, X_comp, threshold=0.5):
        X_comp = self.scaler.transform(X_comp)
        X_comp = self.pca.transform(X_comp)

        probs = self.model.predict(X_comp)
        y_pred = predict_threshold(probs, threshold)
        return y_pred.reshape(-1)


class MLP(TFModel):
    def __init__(self , X, y):

        self.scaler = StandardScaler()
        self.scaler.fit(X)

        layers = [4,6,4]
        params = {
            'patience': 20,
            'epochs': 300,
            'batch_size': 64,
            'val_split': 0.1,
            'pred_th': 0.5
        }
        # Model initialization
        super().__init__(X, y, layers, params)


    def predict(self, X_comp, threshold=0.5):
        X_comp = self.scaler.transform(X_comp)
        probs = self.model.predict(X_comp)
        y_pred = predict_threshold(probs, threshold)
        return y_pred.reshape(-1)


class superMLP(TFModel):
    def __init__(self , X, y):

        self.pca = PCA(random_state=seed, n_components=0.9)
        self.pca.fit(X)
        X_pca = self.pca.transform(X)

        self.models = {'svm': None, 'et': None, 'mlp': None}
        self.models['svm'] = pkl.load(open(f'models/model32.pkl', 'rb'))
        self.models['et'] = pkl.load(open(f'models/model54.pkl', 'rb'))
        self.models['mlp'] = tf.keras.models.load_model('models/model45', custom_objects={'f1_metric': f1_metric})

        for key, model in self.models.items():
            if 'mlp' in key:
                probs = model.predict(self.pca.transform(X)).reshape(-1,1)
            else:
                probs = model.predict_proba(X)[:,1].reshape(-1,1)
            X_pca = np.hstack((X_pca, probs))
            
        layers = [10]
        params = {
            'patience': 15,
            'epochs': 100,
            'batch_size': 16,
            'val_split': 0.1,
            'pred_th': 0.5
        }
        # Model initialization
        super().__init__(X_pca, y, layers, params)


    def predict(self, X_comp, threshold=0.5):
        X_comp_pca = self.pca.transform(X_comp)

        for key, model in self.models.items():
            if 'mlp' in key:
                probs = model.predict(self.pca.transform(X_comp)).reshape(-1,1)
            else:
                probs = model.predict_proba(X_comp)[:,1].reshape(-1,1)
            X_comp_pca = np.hstack((X_comp_pca, probs))

        probs = self.model.predict(X_comp_pca)
        y_pred = predict_threshold(probs, threshold)
        return y_pred.reshape(-1)

        


###################
# Ensembles
###################


class EnsembleModel():

    def __init__(self, models):
        self.basic_models = models
        self.model = None

    def train(self):
        ensemble = []
        for basic_model in self.basic_models:
            name = basic_model.clf.model[0]
            basic_model.train()
            ensemble.append((name, basic_model.model))

        self.model = VotingClassifier(estimators=ensemble, voting='hard')
        return self.model

    def compose(self, names):
        ensemble = []
        for i, basic_model in enumerate(self.basic_models):
            ensemble.append((names[i], basic_model))
        self.model = VotingClassifier(estimators=ensemble, voting='hard')
        return self.model



class Stack():
    def __init__(self):
        self.models = {'svm': None, 'et': None, 'mlp': None}
        self.models['svm'] = pkl.load(open(f'models/model32.pkl', 'rb'))
        self.models['et'] = pkl.load(open(f'models/model54.pkl', 'rb'))
        self.models['mlp'] = tf.keras.models.load_model('models/model45', custom_objects={'f1_metric': f1_metric})
        estimators = [
            ('et', self.models['et'])
        ]
        self.model = StackingClassifier(
            estimators=estimators, final_estimator=self.models['svm']
        )


def VotingHard(sub_list):
    submissions = read_submissions()
    ids = np.arange(0, 1750)
    n_voters = len(sub_list)
    y_pred = []

    for i in ids:
        votes = [submissions[str(v)].target[i] for v in sub_list]
        if sum(votes) > n_voters/2:
            y_pred.append(True)
        else:
            y_pred.append(False)

    return y_pred
