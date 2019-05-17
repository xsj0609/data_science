
import numpy as np
import sys
import time
import pandas as pd

class Dataset(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

class TreeNode(object):
    def __init__(self):
        self.is_leaf = False
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.weight = None

    def _calc_gain(self, G, H, G_l, G_r, H_l, H_r, lambd, gamma):
        """Measure how good a tree is. Equation 7"""
        def calc_term(g, h):
            return np.square(g) / (h + lambd)
        gain = 0.5 * (calc_term(G_l, H_l) +
                      calc_term(G_r, H_r) -
                      calc_term(G, H)) - gamma
        # the bigger gamma, the more convative
        return gain

    def _calc_leaf_weight(self, g, h, lambd):
        """Calc the optimal weight of leaf node. Equation 5"""
        return np.sum(g) / (np.sum(h) + lambd)
    
    def build(self, instances, grad, hessian, eta, depth, param):
        """Algorithm 1"""
        if depth > param['max_depth']:
            # If the depth now is bigger than max depth, it is leaf node, and stop growing.
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * eta
            return
        G = np.sum(grad)
        H = np.sum(hessian)
        best_gain = 0.
        best_feature = None
        best_val = 0.
        best_left_instances = None
        best_right_instances = None
        for feature in range(instances.shape[1]):
            G_l, H_l = 0., 0.
            sorted_instances = instances[:, feature].argsort()
            for j in range(sorted_instances.shape[0]):
                G_l += grad[sorted_instances[j]]
                H_l += hessian[sorted_instances[j]]
                G_r = G - G_l
                H_r = H - H_l
                current_gain = self._calc_gain(G, H, G_l, G_r, H_l, H_r,
                                               param['lambda'], param['gamma'])
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature = feature
                    best_val = instances[sorted_instances[j]][feature]
                    best_left_instances = sorted_instances[:j+1]
                    best_right_instances = sorted_instances[j+1:]
        if best_gain < param['min_split_gain']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * eta
        else:
            self.split_feature = best_feature
            self.split_value = best_val
            self.left_child = TreeNode()
            self.left_child.build(instances[best_left_instances],
                                  grad[best_left_instances],
                                  hessian[best_left_instances],
                                  eta, depth+1, param)

            self.right_child = TreeNode()
            self.right_child.build(instances[best_right_instances],
                                   grad[best_right_instances],
                                   hessian[best_right_instances],
                                   eta, depth+1, param)

    def predict(self, x):
        if self.is_leaf:
            return self.weight
        else:
            if x[self.split_feature] <= self.split_value:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)

class Tree(object):
    """Tree ensemble"""
    def __init__(self):
        self.root = None
    
    def build(self, instances, grad, hessian, eta, param):
        assert len(instances) == len(grad) == len(hessian)
        self.root = TreeNode()
        current_depth = 0
        self.root.build(instances, grad, hessian, eta, current_depth, param)
        
    def predict(self, x):
        return self.root.predict(x)

class GBT(object):
    def __init__(self):
        self.params = {'gamma': 0.,
                       'lambda': 1.,
                       'min_split_gain': 0.1,
                       'max_depth': 5,
                       'learning_rate': 0.3}
        self.best_iteration = None
        
    def _calc_training_data_scores(self, train_set, models):
        if len(models) == 0:
            return None
        X = train_set.X
        scores = np.zeros(len(X))
        for i in range(len(X)):
            scores[i] = self.predict(X[i], models=models)
        return scores
    
    def _calc_l2_gradient(self, train_set, scores):
        labels = train_set.y
        hessian = np.full(len(labels), 2)
        if scores is None:
            grad = np.random.uniform(size=len(labels))
        else:
            grad = np.array([2 * (labels[i] - scores[i]) for i in range(len(labels))])
        return grad, hessian
    
    def _calc_l2_loss(self, models, data_set):
        errors = []
        for x, y in zip(data_set.X, data_set.y):
            errors.append(y - self.predict(x, models))
        return np.mean(np.square(errors))
    
    def _build_learner(self, train_set, grad, hessian, eta):
        learner = Tree()
        learner.build(train_set.X, grad, hessian, eta, self.params)
        return learner
    
    def train(self, params, train_set, valid_set=None, num_boost_rounds=20,
              early_stopping_rounds=5, calc_grad=None, calc_loss=None):
        self.params.update(params)
        models = []
        eta = self.params['learning_rate']
        best_iteration = None
        best_val_loss = np.infty
        start = time.time()
        
        for cnt in range(num_boost_rounds):
            iter_start = time.time()
            scores = self._calc_training_data_scores(train_set, models)
            if calc_grad is None:
                grad, hessian = self._calc_l2_gradient(train_set, scores)
            else:
                grad, hessian = calc_grad(train_set, scores)
            learner = self._build_learner(train_set, grad, hessian, eta)
            models.append(learner)
            if calc_loss is None:
                train_loss = self._calc_l2_loss(models, train_set)
            else:
                train_loss = calc_loss(models, train_set)
            if valid_set is not None:
                if calc_loss is None:
                    val_loss = self._calc_l2_loss(models, valid_set)
                else:
                    val_loss = calc_loss(models, valid_set)
            else:
                val_loss = None
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = cnt
            val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
            print("Iter {:>3}, Train's L2: {:.10f}, Valid's L2: {}, Elapsed: {:.2f} secs"
                  .format(cnt, train_loss, val_loss_str, time.time() - iter_start))
            if cnt - best_iteration >= early_stopping_rounds:
                print('Early stopping, best iteration is: %d' %(best_iteration))
                break
        self.models = models
        self.best_iteration = best_iteration
        print('Train finished. Elapsed: %.2fs, Train Loss: %.2f' %(time.time() - start, train_loss))
        
    def predict(self, x, models=None, num_iter=None):
        if models is None:
            models = self.models
        assert models is not None
        return np.sum([m.predict(x) for m in models[:num_iter]])

'''
data = pd.read_csv('pokemon.csv')
data['Total'] = data['HP'] + data['Defense'] + data['Sp. Atk'] + data['Sp. Def'] + data['Speed'] + data['Attack']

# row_index = np.arange(0,560)
row_index = np.arange(0,20)
cols = ['Total', 'HP', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
y_train = data.loc[row_index, 'Attack'].values
y_test = data.loc[row_index, 'Attack'].values
X_train = data.loc[row_index, cols].values
X_test = data.loc[row_index, cols].values
'''
data = pd.read_csv('TempLinkoping201602.txt')

cols = ['Age','Job','House','Loan']
y_train = data.loc[:, 'y'].values
y_test = data.loc[:, 'y'].values
X_train = data.loc[:, cols].values
X_test = data.loc[:, cols].values

train_data = Dataset(X_train, y_train)
eval_data = Dataset(X_test, y_test)

# pd.DataFrame(data.loc[row_index, cols+['Attack']]).to_csv('pokemon01.csv', index=False, header=True)

params = {}

print('Start training...')
gbt = GBT()
gbt.train(params,
          train_data,
          valid_set=eval_data,
          early_stopping_rounds=5)

y_pred = []
for x in X_test:
    y_pred.append(gbt.predict(x, num_iter=gbt.best_iteration))

y_pred = pd.Series(y_pred)

#y_pred.describe()

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, y_pred) ** 0.5)

#14.639969153776136