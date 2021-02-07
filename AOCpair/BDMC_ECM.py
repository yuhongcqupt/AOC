import time
import matlab
import matlab.engine
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

class BDMC_ECM(object):
    def __init__(self, X_train, y_train, labeled, budget, X_test, y_test):
        self.X = X_train
        self.y = y_train.astype(np.int)
        self.X_test = X_test
        self.y_test = y_test
        #####################
        self.labels = np.unique(y_train)
        self.target = np.array([_ for _ in np.arange(self.labels[0], self.labels[-1], 1)])
        self.N = len(y_train)
        self.labNum = len(self.labels)
        self.budget = deepcopy(budget)
        self.budgetLeft = deepcopy(budget)
        self.recordNote = [i for i in np.arange(1, self.budget + 1, 1)]
        self.initNum = len(labeled)
        ######################
        self.absLabeled = list(deepcopy(labeled))
        self.unLabeled = self.init_unlabeled()
        self.intLabeled = []
        self.Xin = self.init_pool_information()
        ######################
        self.ocModel = self.init_learning_model()
        self.latModel = LogisticAT()
        self.trainIndex = OrderedDict()
        self.trainTarget = OrderedDict()
        ######################
        self.temp_selected = []
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

    def init_learning_model(self):
        model_dict = OrderedDict()
        for tar in self.target:
            model_dict[tar] = LogisticRegression(solver='newton-cg', penalty='l2')
        return model_dict

    def reconstruct_and_train(self):
        self.trainIndex = OrderedDict()
        self.trainTarget = OrderedDict()
        for tar in self.target:
            self.trainIndex[tar] = deepcopy(self.absLabeled)
            self.trainTarget[tar] = deepcopy(self.y[self.absLabeled])
            for j in range(len(self.absLabeled)):
                if self.trainTarget[tar][j] <= tar:
                    self.trainTarget[tar][j] = tar
                else:
                    self.trainTarget[tar][j] = tar + 1
        """add the interval labeled instances into the training set"""
        if self.intLabeled:
            for idx in self.intLabeled:
                for tar in self.target:
                    if tar < self.Xin[idx].inter[0]:
                        self.trainIndex[tar].append(idx)
                        self.trainTarget[tar] = np.append(self.trainTarget[tar], tar + 1)
                    elif tar >= self.Xin[idx].inter[-1]:
                        self.trainIndex[tar].append(idx)
                        self.trainTarget[tar] = np.append(self.trainTarget[tar], tar)
                    else:
                        continue
        for tar in self.target:
            if len(self.trainIndex[tar]) != len(self.trainTarget[tar]):
                print("位置：bdocReConstruct。原因：索引和标记常数不相同")
        for tar, model in self.ocModel.items():
            model.fit(self.X[self.trainIndex[tar]], self.trainTarget[tar])

    def unlabeled_evaluate(self):
        for idx in self.unLabeled:
            len_interval = len(self.Xin[idx].inter)
            if len_interval == 1:
                print("类别归属区间等于1，应该删除")
            elif len_interval == 2:
                tar_single = self.Xin[idx].inter[0]
                prob_dict = OrderedDict()
                prob = self.ocModel[tar_single].predict_proba(self.X[[idx]])[0]
                prob_dict[self.Xin[idx].inter[0]] = prob[0]
                prob_dict[self.Xin[idx].inter[1]] = prob[1]
                self.Xin[idx].prob = prob_dict
                self.Xin[idx].flab = max(prob_dict, key=prob_dict.get)

            elif len_interval >= 3:
                tar_list = [_ for _ in np.arange(self.Xin[idx].inter[0], self.Xin[idx].inter[-1], 1)]
                prob_list = OrderedDict()
                prob_dict = OrderedDict()
                for tar in tar_list:
                    prob_list[tar] = self.ocModel[tar].predict_proba(self.X[[idx]])[0]
                for ele in self.Xin[idx].inter:
                    if ele == self.Xin[idx].inter[0]:
                        prob_dict[ele] = prob_list[ele][0]
                    elif ele == self.Xin[idx].inter[-1]:
                        prob_dict[ele] = prob_list[ele - 1][1]
                    else:
                        prob_dict[ele] = prob_list[ele - 1][1] - prob_list[ele][1]
                self.Xin[idx].prob = prob_dict
                self.Xin[idx].flab = max(prob_dict, key=prob_dict.get)


        for idx in self.intLabeled:
            len_interval = len(self.Xin[idx].inter)
            if len_interval == 1:
                print("类别归属区间等于1，应该删除")
            elif len_interval == 2:
                tar_single = self.Xin[idx].inter[0]
                prob_dict = OrderedDict()
                prob = self.ocModel[tar_single].predict_proba(self.X[[idx]])[0]
                prob_dict[self.Xin[idx].inter[0]] = prob[0]
                prob_dict[self.Xin[idx].inter[1]] = prob[1]
                self.Xin[idx].prob = prob_dict
                self.Xin[idx].flab = max(prob_dict, key=prob_dict.get)

            elif len_interval >= 3:
                tar_list = [_ for _ in np.arange(self.Xin[idx].inter[0], self.Xin[idx].inter[-1], 1)]
                # print("目标区间:", tar_list)
                # print("类别区间:", self.Xin[idx].inter)
                prob_list = OrderedDict()
                prob_dict = OrderedDict()
                for tar in tar_list:
                    prob_list[tar] = self.ocModel[tar].predict_proba(self.X[[idx]])[0]
                for ele in self.Xin[idx].inter:
                    if ele == self.Xin[idx].inter[0]:
                        prob_dict[ele] = prob_list[ele][0]
                    elif ele == self.Xin[idx].inter[-1]:
                        prob_dict[ele] = prob_list[ele - 1][1]
                    else:
                        prob_dict[ele] = prob_list[ele - 1][1] - prob_list[ele][1]
                self.Xin[idx].prob = prob_dict
                self.Xin[idx].flab = max(prob_dict, key=prob_dict.get)

    def instance_selection(self):
        self.temp_selected = []
        metric_list = OrderedDict()
        for idx in self.unLabeled:
            prob = deepcopy(self.Xin[idx].prob)
            ele = max(prob,key=prob.get)
            pmax1 = prob[ele]
            del prob[ele]
            ele = max(prob,key=prob.get)
            pmax2 = prob[ele]
            metric_list[idx] = pmax1 - pmax2
        tar_idx = min(metric_list, key=metric_list.get)
        self.temp_selected.append(tar_idx)
        self.unLabeled.remove(tar_idx)


    def predict(self):
        note = self.budget - self.budgetLeft
        if note not in self.already:
            self.already.append(note)
            self.reconstruct_and_train()
            ###############################
            """直接用二分类分解预测"""
            # proDict = OrderedDict()
            # for tar, model in self.ocModel.items():
            #     proDict[tar] = model.predict_proba(self.X_test)
            # y_pred = np.zeros(len(self.y_test))
            # for j in range(len(self.y_test)):
            #     prob = OrderedDict()
            #     for ele in self.labels:
            #         if ele == self.labels[0]:
            #             prob[ele] = proDict[ele][j][0]
            #         elif ele == self.labels[-1]:
            #             prob[ele] = proDict[ele - 1][j][1]
            #         else:
            #             prob[ele] = proDict[ele - 1][j][1] - proDict[ele][j][1]
            #     y_pred[j] = max(prob, key=prob.get)
            ###############################
            self.latModel.fit(self.X[self.absLabeled],self.y[self.absLabeled])
            y_pred = self.latModel.predict(self.X_test)
            Acc = accuracy_score(y_true=self.y_test, y_pred=y_pred)
            Recall = recall_score(y_true=self.y_test, y_pred=y_pred, average='macro')
            F1 = f1_score(y_true=self.y_test, y_pred=y_pred, average='macro')
            MAE = mean_absolute_error(y_true=self.y_test, y_pred=y_pred)
            self.AccList.append(Acc)
            self.RecallList.append(Recall)
            self.F1List.append(F1)
            self.MAEList.append(MAE)
            self.QYRList.append((len(self.absLabeled) - self.initNum) / (self.budget - self.budgetLeft))
            self.ALC_Acc += Acc
            self.ALC_Recall += Recall
            self.ALC_F1 += F1
            self.ALC_MAE += MAE


    def construct_cost_pair(self):
        print("                 ")
        print("                 ")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>进入匹配环节<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        if self.intLabeled:
            for idx in self.intLabeled:
                print("区间样本{}的真实标记为{}概率估计{}".format(idx,self.y[idx],self.Xin[idx].prob))
                if len(self.Xin[idx].inter) <= 3:
                    self.Xin[idx].evalab = self.Xin[idx].inter[1]
                    print("区间样本{}的匹配{}".format(idx, self.Xin[idx].evalab))
                else:
                    cost = OrderedDict()
                    for lab in self.Xin[idx].inter[1:-1]:
                        same_cost = self.Xin[idx].prob[lab] * 1
                        left_cost = deepcopy(same_cost)
                        right_cost = deepcopy(same_cost)
                        left_price = np.floor(np.log2(lab - self.Xin[idx].inter[0]))
                        right_price = np.floor(np.log2(self.Xin[idx].inter[-1] - lab))
                        for ele, pro in self.Xin[idx].prob.items():
                            if ele != lab:
                                left_cost += pro * left_price
                            else:
                                break
                        for ele, pro in self.Xin[idx].prob.items():
                            if ele > lab:
                                right_cost += pro * right_price
                            else:
                                continue
                        cost[lab] = same_cost + left_cost + right_cost
                    print("区间样本{}的代价{}".format(idx,cost))
                    self.Xin[idx].evalab = min(cost,key=cost.get)
                    print("区间样本{}的匹配{}".format(idx, self.Xin[idx].evalab))

        for idx in self.temp_selected:
            print("新选样本{}的真实标记为{}概率估计{}".format(idx,self.y[idx],self.Xin[idx].prob))
            if len(self.Xin[idx].inter) <= 3:
                self.Xin[idx].evalab = self.Xin[idx].inter[1]
                print("新选样本{}的匹配{}".format(idx, self.Xin[idx].evalab))
            else:
                cost = OrderedDict()
                for lab in self.Xin[idx].inter[1:-1]:
                    same_cost = self.Xin[idx].prob[lab] * 1
                    left_cost = 0
                    right_cost = 0
                    left_price = np.floor(np.log2(lab - self.Xin[idx].inter[0]))
                    right_price = np.floor(np.log2(self.Xin[idx].inter[-1] - lab))
                    for ele, pro in self.Xin[idx].prob.items():
                        if ele != lab:
                            left_cost += pro * 1
                        else:
                            break
                    for ele, pro in self.Xin[idx].prob.items():
                        if ele != lab:
                            left_cost += pro * left_price
                        else:
                            break
                    for ele, pro in self.Xin[idx].prob.items():
                        if ele > lab:
                            right_cost += pro * 1
                        else:
                            continue
                    for ele, pro in self.Xin[idx].prob.items():
                        if ele > lab:
                            right_cost += pro * right_price
                        else:
                            continue
                    cost[lab] = same_cost + left_cost + right_cost
                print("新选样本{}的代价{}".format(idx, cost))
                self.Xin[idx].evalab = min(cost, key=cost.get)
                print("新选样本{}的匹配{}".format(idx, self.Xin[idx].evalab))

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
                print("处理区间标记样本{},区间为{},匹配标记为{},真实标记为{}".format(idx, self.Xin[idx].inter, self.Xin[idx].evalab,self.y[idx]))
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
                print("处理新选标记样本{},区间为{},匹配标记为{},真实标记为{}".format(idx, self.Xin[idx].inter, self.Xin[idx].evalab,self.y[idx]))
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
            self.reconstruct_and_train()
            self.unlabeled_evaluate()
            self.instance_selection()
            self.construct_cost_pair()
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
    p = Path("D:\datasets")
    names = ["Housing"]


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
            SKF = StratifiedKFold(n_splits=5, shuffle=True)
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
                model = BDMC_ECM(X_train=train_X, y_train=train_y, labeled=labeled, budget=Budget, X_test=test_X,
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
