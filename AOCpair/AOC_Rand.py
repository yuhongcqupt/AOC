import time
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, recall_score
from mord import LogisticAT

class Struct(object):
    def __init__(self,idx):
        self.inter = None
        self.prob = OrderedDict()
        self.flab = None
        self.slab = None
        self.evalab = None

class AORrandom(object):
    def __init__(self,X_train, y_train, labeled, budget, X_test, y_test,):
        self.X = X_train
        self.y = y_train.astype(np.int)
        self.X_test = X_test
        self.y_test = y_test.astype(np.int)
        self.labels = np.unique(y_train)
        self.target = np.array([_ for _ in np.arange(self.labels[0], self.labels[-1], 1)])
        self.N = len(y_train)
        self.labNum = len(self.labels)
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)
        self.recordNote = [i for i in np.arange(1, self.budget + 1, 1)]
        self.initNum = len(labeled)
        ##############################
        self.absLabeled = list(deepcopy(labeled))
        self.unLabeled = self.init_unlabeled()
        self.intLabeled = []
        self.Xin = self.init_pool_information()
        ##############################
        self.latModel = LogisticAT()
        self.latModel.fit(self.X[self.absLabeled],self.y[self.absLabeled])
        self.theta, self.w = self.latModel.theta_w()
        self.temp_selected = []
        self.struct_set = OrderedDict()
        self.sameSet = OrderedDict()
        self.chainSet = OrderedDict()
        self.t = int(0)
        self.n_theta = int(len(self.labels)-1)
        ##############################
        self.certainList = []
        self.uncertainList = []
        self.uncertainPool = []
        self.already = []
        '''Assessment metrics'''
        self.AccList = []
        self.RecallList = []
        self.F1List = []
        self.MAEList = []
        self.QYRList = []
        self.numABS = 0
        '''Additional assessment metrics'''
        self.ALC_Acc = 0
        self.ALC_Recall = 0
        self.ALC_F1 = 0
        self.ALC_MAE = 0

    def dist(self, a, b):
        return np.sqrt(sum((a - b) ** 2))

    def init_unlabeled(self):
        unlabeled = [_ for _ in range(self.N)]
        for ele in self.absLabeled:
            unlabeled.remove(ele)
        return unlabeled

    def init_pool_information(self):
        Xin = OrderedDict()
        for idx in self.unLabeled:
            Xin[idx] = Struct(idx=idx)
            Xin[idx].inter = list(self.labels)
        return Xin

    def reTrain(self):
        self.latModel.fit(X=self.X[self.absLabeled],y=self.y[self.absLabeled])
        self.theta, self.w = self.latModel.theta_w()

    def unlabeled_evaluate(self):
        if self.unLabeled:
            prob_matrix = self.latModel.predict_proba(self.X[self.unLabeled])
            for j, idx in enumerate(self.unLabeled):
                prob = OrderedDict()
                for r, lab in enumerate(self.labels):
                    prob[lab] = prob_matrix[j][r]
                self.Xin[idx].prob = prob
                self.Xin[idx].flab = max(prob,key=prob.get)
        if self.intLabeled:
            for idx in self.intLabeled:
                interval = deepcopy(self.Xin[idx].inter)
                # theta_idx = []
                # for lab in interval:
                #     theta_idx.append(self.labels[lab])
                traindx = []
                for jdx in self.absLabeled:
                    if self.y[jdx] in interval:
                        traindx.append(jdx)
                model = LogisticAT(alpha=1.0, verbose=0, max_iter=1000)
                model.fit(self.X[traindx], self.y[traindx])
                prob_list = model.predict_proba(self.X[[idx]])[0]
                self.Xin[idx].prob = OrderedDict()
                for r, lab in enumerate(self.Xin[idx].inter):
                    self.Xin[idx].prob[lab] = prob_list[r]
                self.Xin[idx].flab = max(self.Xin[idx].prob, key=self.Xin[idx].prob.get)

    def instance_selection(self):
        self.temp_selected = []
        while len(self.temp_selected) < 1:
            unlab_project = self.X[self.unLabeled].dot(self.w)
            tmp = self.theta[self.t] - np.asarray(unlab_project)
            if self.t < self.n_theta-1:
                self.t += 1
            else:
                self.t = 0
            ordjdx = np.argsort(abs(tmp))
            idx = self.unLabeled[ordjdx[0]]
            #####################
            # i = 0
            # idx = self.unLabeled[ordjdx[i]]
            # while idx in self.intLabeled or idx in self.uncertainPool or idx in self.absLabeled:
            #     i += 1
            #     idx = self.unLabeled[ordjdx[i]]
            #####################
            self.temp_selected.append(idx)
            self.unLabeled.remove(idx)

    def predict(self):
        #################使用单分类器
        note = self.budget - self.budgetLeft
        if note not in self.already:
            self.already.append(note)
            y_pred = self.latModel.predict(self.X_test)
            self.AccList.append(accuracy_score(y_true=self.y_test, y_pred=y_pred))
            self.RecallList.append(recall_score(y_true=self.y_test, y_pred=y_pred, average='macro'))
            self.F1List.append(f1_score(y_true=self.y_test, y_pred=y_pred, average='macro'))
            self.MAEList.append(mean_absolute_error(y_true=self.y_test, y_pred=y_pred))
            self.QYRList.append((len(self.absLabeled) - self.initNum) / (self.budget - self.budgetLeft))
            self.ALC_Acc += accuracy_score(y_true=self.y_test, y_pred=y_pred)
            self.ALC_Recall += recall_score(y_true=self.y_test, y_pred=y_pred, average='macro')
            self.ALC_F1 += f1_score(y_true=self.y_test, y_pred=y_pred, average='macro')
            self.ALC_MAE += mean_absolute_error(y_true=self.y_test, y_pred=y_pred)

    def contruct_rand_pair(self):
        if self.intLabeled:
            for idx in self.intLabeled:
                if len(self.Xin[idx].inter) >= 3:
                    self.Xin[idx].evalab = np.random.choice(self.Xin[idx].inter[1:-1], 1, replace=False)[0]
                elif len(self.Xin[idx].inter) == 2:
                    self.Xin[idx].evalab = np.random.choice(self.Xin[idx].inter, 1, replace=False)[0]
        for idx in self.temp_selected:
            self.Xin[idx].evalab = np.random.choice(self.Xin[idx].inter[1:-1], 1, replace=False)[0]

    def query_reasoning(self):
        if self.labNum <= 3:
            for idx in self.temp_selected:
                if self.budgetLeft <= 0:
                    break
                else:
                    self.budgetLeft -= 1
                    self.absLabeled.append(idx)
                    if (self.budget - self.budgetLeft) in self.recordNote:  ### 记录中途分类器的分类结果
                        self.predict()
        else:
            for idx in self.intLabeled:
                print("处理区间标记样本{},区间为{},匹配标记为{}".format(idx, self.Xin[idx].inter, self.Xin[idx].evalab))
                if self.budgetLeft <= 0:
                    break
                else:
                    self.budgetLeft -= 1
                    if self.y[idx] == self.Xin[idx].evalab:
                        self.absLabeled.append(idx)
                        self.intLabeled.remove(idx)
                    elif self.y[idx] < self.Xin[idx].evalab:
                        if self.Xin[idx].evalab - self.Xin[idx].inter[0] == 1:
                            self.absLabeled.append(idx)
                            self.intLabeled.remove(idx)
                        elif self.Xin[idx].evalab - self.Xin[idx].inter[0] > 1:
                            self.Xin[idx].inter = [_ for _ in
                                                   np.arange(self.Xin[idx].inter[0], self.Xin[idx].evalab, 1)]
                    elif self.y[idx] > self.Xin[idx].evalab:
                        if self.Xin[idx].inter[-1] - self.Xin[idx].evalab == 1:
                            self.absLabeled.append(idx)
                            self.intLabeled.remove(idx)
                        elif self.Xin[idx].inter[-1] - self.Xin[idx].evalab > 1:
                            self.Xin[idx].inter = [_ for _ in
                                                   np.arange(self.Xin[idx].evalab + 1, self.Xin[idx].inter[-1] + 1, 1)]
                    if (self.budget - self.budgetLeft) in self.recordNote:  ### 记录中途分类器的分类结果
                        self.predict()

            for idx in self.temp_selected:
                print("处理新选标记样本{},区间为{},匹配标记为{}".format(idx, self.Xin[idx].inter, self.Xin[idx].evalab))
                if self.budgetLeft <= 0:
                    break
                else:
                    self.budgetLeft -= 1
                    if self.y[idx] == self.Xin[idx].evalab:
                        self.absLabeled.append(idx)
                    elif self.y[idx] < self.Xin[idx].evalab:
                        if self.Xin[idx].evalab - self.Xin[idx].inter[0] == 1:
                            self.absLabeled.append(idx)
                        elif self.Xin[idx].evalab - self.Xin[idx].inter[0] > 1:
                            self.Xin[idx].inter = [_ for _ in
                                                   np.arange(self.Xin[idx].inter[0], self.Xin[idx].evalab, 1)]
                            self.intLabeled.append(idx)
                    elif self.y[idx] > self.Xin[idx].evalab:
                        if self.Xin[idx].inter[-1] - self.Xin[idx].evalab == 1:
                            self.absLabeled.append(idx)
                        elif self.Xin[idx].inter[-1] - self.Xin[idx].evalab > 1:
                            self.Xin[idx].inter = [_ for _ in
                                                   np.arange(self.Xin[idx].evalab + 1, self.Xin[idx].inter[-1] + 1, 1)]
                            self.intLabeled.append(idx)
                    if (self.budget - self.budgetLeft) in self.recordNote:  ### 记录中途分类器的分类结果
                        self.predict()


    def start(self):
        while self.budgetLeft > 0:
            self.reTrain()
            self.unlabeled_evaluate()
            self.instance_selection()
            self.contruct_rand_pair()
            self.query_reasoning()



if __name__ == '__main__':

    s_time = time.time()
    class Stores1(object):
        def __init__(self):
            self.Acclist = []
            self.F1list = []
            self.MAElist = []
            self.QYRlist = []
            self.ALC_Acc = []
            self.ALC_F1 = []
            self.ALC_MAE = []


    class Stores2(object):
        def __init__(self):
            self.ACC_mean = []
            self.ACC_std = []
            self.F1_mean = []
            self.F1_std = []
            self.MAE_mean = []
            self.MAE_std = []
            self.QYR_mean = []
            self.QYR_std = []
            self.ALC_Acc_mean = []
            self.ALC_Acc_std = []
            self.ALC_F1_mean = []
            self.ALC_F1_std = []
            self.ALC_MAE_mean = []
            self.ALC_MAE_std = []


    #######################数据集###############################
    p = Path("D:\OCdata")
    names = ["LEV"]


    for name in names:
        path = p.joinpath(name + ".csv")
        print("#####################################################{}".format(path))
        data = np.array(pd.read_csv(path, header=None))
        X = data[:, :-1]
        y = data[:, -1]
        Rounds = 1
        labNum = len(np.unique(y))
        print("数据集信息{}".format(set(y)))
        budgetlist = np.array([labNum * i for i in range(1, 21)])
        Budget = labNum * 20
        ###------------------------------
        # store = OrderedDict()
        # for method in Methods:
        #     store[method] = Stores1()
        # sttoo = OrderedDict()
        # for method in Methods:
        #     sttoo[method] = Stores2()

        for r in range(Rounds):
            SKF = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)
            for train_idx, test_idx in SKF.split(X, y):
                # print("类别个数=",labNum,"训练数据=",len(train_idx),"测试数据=",len(test_idx))
                train_X = X[train_idx]
                train_y = y[train_idx]
                test_X = X[test_idx]
                test_y = y[test_idx]
                labeled = []
                label_dict = OrderedDict()
                for lab in np.unique(train_y):
                    label_dict[lab] = []
                for idx in range(len(train_y)):
                    label_dict[train_y[idx]].append(idx)
                for idxlist in label_dict.values():
                    for jdx in np.random.choice(idxlist, size=1, replace=False):
                        labeled.append(jdx)
                model = AORrandom(X_train=train_X, y_train=train_y, labeled=labeled, budget=Budget, X_test=test_X,
                               y_test=test_y)
                model.start()
                print("精度", len(model.AccList), model.AccList)
                print("F1", len(model.F1List), model.F1List)
                print("MAE", len(model.MAEList), model.MAEList)
                print("QYR", len(model.QYRList), model.QYRList)
                print("ALC_Acc", model.ALC_Acc)
                print("ALC_F1", model.ALC_F1)
                print("ALC_MAE", model.ALC_MAE)
                print("剩余标记预算", model.budgetLeft)
                e_time = time.time()
                print("消耗时间=",e_time - s_time)
                break
            break

        break

