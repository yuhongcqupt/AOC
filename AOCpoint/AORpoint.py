'''
author:Deniu He
date:2020-12-11
organization: CQUPT
Reference: D. Wu, C. Lin, J. Huang. Active learning for regression using greeding sampling. Information Sciences, 2019, 474: 90-105.
'''
import xlwt
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import accuracy_score, mean_absolute_error
from collections import OrderedDict
from mord import LogisticAT
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from sklearn.cluster import KMeans
from skactiveml.classifier._pwc import PWC
from skactiveml.pool._probal import McPAL



class AORpoint():
    def __init__(self,X_pool,y_pool,labeled,budget,X_test,y_test):
        self.X_pool = X_pool
        self.y_pool = y_pool.astype(np.int)
        self.X_test = X_test
        self.y_test = y_test.astype(np.int)
        self.labeled = list(deepcopy(labeled))
        self.model = LogisticAT()
        self.theta = None
        self.w = None
        self.unlabeled = self.initialization()
        self.budgetLeft = deepcopy(budget)
        self.budget = deepcopy(budget)
        self.AccList = []
        self.MAEList = []
        self.AUCList = []
        self.ALC_Acc = 0.0
        self.ALC_MAE = 0.0
        self.ALC_Acc_10k = 0.0
        self.ALC_MAE_10k = 0.0

    def initialization(self):
        unlabeled = [i for i in range(len(self.y_pool))]
        for j in self.labeled:
            unlabeled.remove(j)
        self.model.fit(X=self.X_pool[self.labeled], y=self.y_pool[self.labeled])
        self.theta, self.w = self.model.theta_w()
        return unlabeled

    def select_1(self):
        n_theta = len(set(self.y_pool)) - 1
        while self.budgetLeft > 0:
            for i in range(n_theta):
                unlab_project = self.X_pool[self.unlabeled].dot(self.w)
                tmp = self.theta[:, None] - np.asarray(unlab_project)
                ordjdx = np.argsort(abs(tmp[i]))
                tar_idx = self.unlabeled[ordjdx[0]]
                self.labeled.append(tar_idx)
                self.unlabeled.remove(tar_idx)
                self.budgetLeft -= 1
                self.model.fit(X=self.X_pool[self.labeled], y=self.y_pool[self.labeled])
                self.theta, self.w = self.model.theta_w()
            Acc = accuracy_score(y_true=self.y_test, y_pred=self.model.predict(self.X_test))
            MAE = mean_absolute_error(y_true=self.y_test, y_pred=self.model.predict(self.X_test))
            self.AccList.append(Acc)
            self.MAEList.append(MAE)
            self.ALC_Acc += Acc
            self.ALC_MAE += MAE

    def select(self):
        n_theta = len(set(self.y_pool)) - 1
        while self.budgetLeft > 0:
            for t in range(n_theta):
                if self.budgetLeft <= 0:
                    break
                unlab_project = self.X_pool[self.unlabeled].dot(self.w)
                tmp = self.theta[t] - np.asarray(unlab_project)
                ordjdx = np.argsort(abs(tmp))
                tar_idx = self.unlabeled[ordjdx[0]]
                self.labeled.append(tar_idx)
                self.unlabeled.remove(tar_idx)
                self.budgetLeft -= 1
                self.model.fit(X=self.X_pool[self.labeled], y=self.y_pool[self.labeled])
                self.theta, self.w = self.model.theta_w()
                Acc = accuracy_score(y_true=self.y_test, y_pred=self.model.predict(self.X_test))
                MAE = mean_absolute_error(y_true=self.y_test, y_pred=self.model.predict(self.X_test))
                self.AccList.append(Acc)
                self.MAEList.append(MAE)
                self.ALC_Acc += Acc
                self.ALC_MAE += MAE
                if self.budgetLeft >= 0.5 * self.budget:
                    self.ALC_Acc_10k += Acc
                    self.ALC_MAE_10k += MAE



if __name__ == '__main__':
    p = Path("D:\KDD")
    names = ["SWD"]

    method = ["AOCpoint"]

    class result():
        def __init__(self):
            self.AccList = []
            self.MAEList = []
            self.ALC_Acc = []
            self.ALC_MAE = []
            self.ALC_Acc_10k = []
            self.ALC_MAE_10k = []

    class Stores():
        def __init__(self):
            self.AccList_mean = []
            self.AccList_std = []
            self.AccList_max = []
            self.AccList_min = []
            self.MAEList_mean = []
            self.MAEList_std = []
            self.MAEList_max = []
            self.MAEList_min = []
            self.ALC_Acc_mean = []
            self.ALC_Acc_std = []
            self.ALC_Acc_max = []
            self.ALC_Acc_min = []
            self.ALC_MAE_mean = []
            self.ALC_MAE_std = []
            self.ALC_MAE_max = []
            self.ALC_MAE_min = []

            self.ALC_Acc_mean_10k = []
            self.ALC_Acc_std_10k = []
            self.ALC_Acc_max_10k = []
            self.ALC_Acc_min_10k = []
            self.ALC_MAE_mean_10k = []
            self.ALC_MAE_std_10k = []
            self.ALC_MAE_max_10k = []
            self.ALC_MAE_min_10k = []


    for name in names:
        path = p.joinpath(name + ".csv")
        print("#####################################################{}".format(path))
        data = np.array(pd.read_csv(path, header=None))
        X = data[:, :-1]
        y = data[:, -1]
        labNum = len(set(y))
        Budget = 10 * labNum
        Rounds = 5
        RESULT = result()
        STORE = Stores()
        for r in range(Rounds):
            SKF = StratifiedKFold(n_splits=5, shuffle=True)
            for train_idx, test_idx in SKF.split(X, y):
                print("size of datasets=",labNum,"size of pool set=",len(train_idx),"size of testing set=",len(test_idx))
                train_X = X[train_idx]
                train_y = y[train_idx].astype(np.int)
                test_X = X[test_idx]
                test_y = y[test_idx]
                labeled = []
                label_dict = OrderedDict()
                for lab in np.unique(train_y):
                    label_dict[lab] = []
                for idx in range(len(train_y)):
                    label_dict[train_y[idx]].append(idx)
                for idxlist in label_dict.values():
                    for jdx in np.random.choice(idxlist,size=1,replace=False):
                        labeled.append(jdx)
                print("Conduct AORpoint")
                ALmodel = AORpoint(X_pool=train_X,y_pool=train_y,labeled=labeled,budget=Budget,X_test=test_X,y_test=test_y)
                ALmodel.select()
                RESULT.AccList.append(ALmodel.AccList)
                RESULT.MAEList.append(ALmodel.MAEList)
                RESULT.ALC_Acc.append(ALmodel.ALC_Acc)
                RESULT.ALC_MAE.append(ALmodel.ALC_MAE)
                RESULT.ALC_Acc_10k.append(ALmodel.ALC_Acc_10k)
                RESULT.ALC_MAE_10k.append(ALmodel.ALC_MAE_10k)

        STORE.AccList_mean = np.mean(RESULT.AccList,axis=0)
        STORE.AccList_std = np.std(RESULT.AccList,axis=0)
        STORE.AccList_max = np.max(RESULT.AccList,axis=0)
        STORE.AccList_min = np.min(RESULT.AccList,axis=0)

        STORE.MAEList_mean = np.mean(RESULT.MAEList,axis=0)
        STORE.MAEList_std = np.std(RESULT.MAEList,axis=0)
        STORE.MAEList_max = np.max(RESULT.MAEList,axis=0)
        STORE.MAEList_min = np.min(RESULT.MAEList,axis=0)

        STORE.ALC_Acc_mean = np.mean(RESULT.ALC_Acc)
        STORE.ALC_Acc_std = np.std(RESULT.ALC_Acc)
        STORE.ALC_Acc_max = np.max(RESULT.ALC_Acc)
        STORE.ALC_Acc_min = np.min(RESULT.ALC_Acc)

        STORE.ALC_MAE_mean = np.mean(RESULT.ALC_MAE)
        STORE.ALC_MAE_std = np.std(RESULT.ALC_MAE)
        STORE.ALC_MAE_max = np.max(RESULT.ALC_MAE)
        STORE.ALC_MAE_min = np.min(RESULT.ALC_MAE)

        STORE.ALC_Acc_mean_10k = np.mean(RESULT.ALC_Acc_10k)
        STORE.ALC_Acc_std_10k = np.std(RESULT.ALC_Acc_10k)
        STORE.ALC_Acc_max_10k = np.max(RESULT.ALC_Acc_10k)
        STORE.ALC_Acc_min_10k = np.min(RESULT.ALC_Acc_10k)

        STORE.ALC_MAE_mean_10k = np.mean(RESULT.ALC_MAE_10k)
        STORE.ALC_MAE_std_10k = np.std(RESULT.ALC_MAE_10k)
        STORE.ALC_MAE_max_10k = np.max(RESULT.ALC_MAE_10k)
        STORE.ALC_MAE_min_10k = np.min(RESULT.ALC_MAE_10k)

        sheet_names = ["Acc_mean", "Acc_std", "Acc_max", "Acc_min", "MAE_mean","MAE_std", "MAE_max", "MAE_min", "ALC_Acc", "ALC_MAE","ALC_Acc_10k", "ALC_MAE_10k"]
        save_path = Path(r"D:\AOC_experiment\pointwise\nAOR")
        workbook = xlwt.Workbook()
        for sn in sheet_names:
            sheet = workbook.add_sheet(sn)
            if sn == "Acc_mean":
                sheet.write(0, 0, method[0])
                for j in range(1, len(STORE.AccList_mean)+1):
                    sheet.write(0,j,STORE.AccList_mean[j-1])
            elif sn == "Acc_std":
                sheet.write(0, 0, method[0])
                for j in range(1, len(STORE.AccList_std)+1):
                    sheet.write(0,j,STORE.AccList_std[j-1])
            elif sn == "Acc_max":
                sheet.write(0, 0, method[0])
                for j in range(1, len(STORE.AccList_max)+1):
                    sheet.write(0,j,STORE.AccList_max[j-1])
            elif sn == "Acc_min":
                sheet.write(0, 0, method[0])
                for j in range(1, len(STORE.AccList_min)+1):
                    sheet.write(0,j,STORE.AccList_min[j-1])
            elif sn == "MAE_mean":
                sheet.write(0, 0, method[0])
                for j in range(1, len(STORE.MAEList_mean)+1):
                    sheet.write(0,j,STORE.MAEList_mean[j-1])
            elif sn == "MAE_std":
                sheet.write(0, 0, method[0])
                for j in range(1, len(STORE.MAEList_std)+1):
                    sheet.write(0,j,STORE.MAEList_std[j-1])
            elif sn == "MAE_max":
                sheet.write(0, 0, method[0])
                for j in range(1, len(STORE.MAEList_max)+1):
                    sheet.write(0,j,STORE.MAEList_max[j-1])
            elif sn == "MAE_min":
                sheet.write(0, 0, method[0])
                for j in range(1, len(STORE.MAEList_min)+1):
                    sheet.write(0,j,STORE.MAEList_min[j-1])
            elif sn == "ALC_Acc":
                sheet.write(0, 0, method[0])
                sheet.write(0, 1, STORE.ALC_Acc_mean)
                sheet.write(0, 2, STORE.ALC_Acc_std)
                sheet.write(0, 3, STORE.ALC_Acc_max)
                sheet.write(0, 4, STORE.ALC_Acc_min)
            elif sn == "ALC_MAE":
                sheet.write(0, 0, method[0])
                sheet.write(0, 1, STORE.ALC_MAE_mean)
                sheet.write(0, 2, STORE.ALC_MAE_std)
                sheet.write(0, 3, STORE.ALC_MAE_max)
                sheet.write(0, 4, STORE.ALC_MAE_min)
            elif sn == "ALC_Acc_10k":
                sheet.write(0, 0, method[0])
                sheet.write(0, 1, STORE.ALC_Acc_mean_10k)
                sheet.write(0, 2, STORE.ALC_Acc_std_10k)
                sheet.write(0, 3, STORE.ALC_Acc_max_10k)
                sheet.write(0, 4, STORE.ALC_Acc_min_10k)
            elif sn == "ALC_MAE_10k":
                sheet.write(0, 0, method[0])
                sheet.write(0, 1, STORE.ALC_MAE_mean_10k)
                sheet.write(0, 2, STORE.ALC_MAE_std_10k)
                sheet.write(0, 3, STORE.ALC_MAE_max_10k)
                sheet.write(0, 4, STORE.ALC_MAE_min_10k)
        save_path = str(save_path.joinpath(name + "-AORpoint"+".xls"))
        workbook.save(save_path)
