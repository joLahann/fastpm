
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/001_PPP_Models.ipynb

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

class PPModel():
    def predict(self,test,index=2):
        nsp=self.next_step_prediction(test,index)
        nrp=self.next_resource_prediction(test,index)
        lrp=self.last_resource_prediction(test,index)
        op=self.outcome_prediction(test,index)
        dtnep=self.duration_to_next_event_prediction(test,index)
        dtep=self.duration_to_end_prediction(test,index)
        asp=self.activity_suffix_prediction(test,index)
        rsp=self.resource_suffix_prediction(test,index)
        return nsp, nrp, lrp, op, dtnep, dtep, asp, rsp

    def _return_results(self, index, col):
        tmp = self.cases.apply(lambda x: x[col].values[index:-1])
        tmp = tmp[tmp.apply(lambda x: len(x)>0)]
        return tmp

    def name(self): return self.__class__.__name__
    def train(self,data):pass
    def next_step_prediction(self,test,index=2): return pd.Series(dtype=float)
    def next_resource_prediction(self,test,index=2): return pd.Series(dtype=float)
    def last_resource_prediction(self,test,index=2): return pd.Series(dtype=float)
    def outcome_prediction(self,test,index=2):  return pd.Series(dtype=float)
    def duration_to_next_event_prediction(self,test,index=2): return pd.Series(dtype=float)
    def duration_to_end_prediction(self,test,index=2):  return pd.Series(dtype=float)
    def activity_suffix_prediction(self,test,index=2):  return pd.Series(dtype=float)
    def resource_suffix_prediction(self,test,index=2):  return pd.Series(dtype=float)

    def reset(self): pass

def evaluate_outcome_prediction(pred,ground_truth): return flat_accuracy(pred,ground_truth)
def flat_accuracy(pred,ground_truth):
    x=np.array([j for i in pred.values for j in i])
    y=np.array([j for i in ground_truth.values for j in i])
    return float((x==y).mean()) if len(x)==len(y) else None

def evaluate_next_step_prediction(pred,ground_truth):
    #print(len(pred[0]), len(ground_truth[0]))
    #print(pred[0], ground_truth[0])
    return flat_accuracy(pred,ground_truth)
def evaluate_next_resource_prediction(pred,ground_truth):
    #print(len(pred[0]), len(ground_truth[0]))
    #print(pred[0], ground_truth[0])
    return flat_accuracy(pred,ground_truth)
def evaluate_last_resource_prediction(pred,ground_truth): return flat_accuracy(pred,ground_truth)
def dl_measure(pred,ground_truth):
    pred=pred.values
    ground_truth=ground_truth.values
    scores=[]
    if len(pred)!=len(ground_truth): return None
    for i in range(len(pred)):
        if len(pred[i])!=len(ground_truth[i]): return None
        for j in range(len(pred[i])):
            scores.append(1-normalized_damerau_levenshtein_distance(pred[i][j], ground_truth[i][j]))
    return np.mean(scores)

def evaluate_activity_suffix_prediction(pred,ground_truth): return dl_measure(pred,ground_truth) # 1
def evaluate_resource_suffix_prediction(pred,ground_truth): return dl_measure(pred,ground_truth) # 1

def mse(output, targ):
    if not torch.is_tensor(output): output=tensor(output)
    if not torch.is_tensor(targ): targ=tensor(targ)
    return (output.squeeze(-1) - targ).pow(2).mean()

def flat_mse(pred,ground_truth):
    x=np.array([j for i in pred.values for j in i])
    y=np.array([j for i in ground_truth.values for j in i])
    return float(mse(x,y)) if len(x)==len(y) else None

def evaluate_duration_to_next_event_prediction(pred,ground_truth): return flat_mse(pred,ground_truth)
def evaluate_duration_to_end_prediction(pred,ground_truth): return flat_mse(pred,ground_truth)

class Perfect_PPM(PPModel):
    def predict(self,test,index=2):
        self.cases=test.groupby('trace_id')
        return super().predict(test,index)

    def _return_ground_truth(self,test,index,col):
        tmp=self.cases.apply(lambda x: x[col].values[index:-1])
        tmp=tmp[tmp.apply(lambda x: len(x)!=0)]
        return tmp

    def next_step_prediction(self,test,index=2): return self._return_ground_truth(test,index,'next activity')

    def next_resource_prediction(self,test,index=2): return self._return_ground_truth(test,index,'next resource')

    def last_resource_prediction(self,test,index=2): return self._return_ground_truth(test,index,'last resource')

    def outcome_prediction(self,test,index=2):  return self._return_ground_truth(test,index,'outcome')

    def duration_to_next_event_prediction(self,test,index=2): return self._return_ground_truth(test,index,'next relative timestamp')

    def duration_to_end_prediction(self,test,index=2): return self._return_ground_truth(test,index,'duration to end')

    def activity_suffix_prediction(self,test,index=2): return self._return_ground_truth(test,index,'activity suffix')

    def resource_suffix_prediction(self,test,index=2): return self._return_ground_truth(test,index,'resource suffix')

class MFV_PPM(PPModel):
    def predict(self,test,index=2):
        self.result_df=test.copy()
        mfa=test['activity'].value_counts().index[0]
        mfr=test['resource'].value_counts().index[0]
        self.result_df['mfa']=mfa
        self.result_df['mfr']=mfr


        self.cases=self.result_df.groupby('trace_id')
        self.median_group_len=int(self.cases.size().median())
        self.result_df['mfa suffix']=[[mfa for i in range(self.median_group_len-index)] for k in range(len(test))]
        self.result_df['mfr suffix']=[[mfr for i in range(self.median_group_len-index)] for k in range(len(test))]
        return super().predict(test,index)

    #def _return_results(self,index,col):
    #    tmp=self.cases.apply(lambda x: x[col].values[index:-1])
    #    tmp=tmp[tmp.apply(lambda x: len(x)>0)]
    #    return tmp

    def next_step_prediction(self,test,index=2): return PPModel._return_results(self, index, 'mfa')

    def next_resource_prediction(self,test,index=2): return PPModel._return_results(self, index, 'mfr')

    def last_resource_prediction(self,test,index=2): return PPModel._return_results(self, index, 'mfr')

    def outcome_prediction(self,test,index=2):  return PPModel._return_results(self, index, 'mfa')
  # Most Frequent Value Model
    def activity_suffix_prediction(self,test,index=2): return PPModel._return_results(self, index, 'mfa suffix')

    def resource_suffix_prediction(self,test,index=2): return PPModel._return_results(self, index, 'mfr suffix')

class RandomForest_PPM(PPModel):
    def __init__(self):
        self.n_estimators = 10
        self.clf1 = RandomForestClassifier(n_estimators=self.n_estimators)
        self.clf2 = RandomForestClassifier(n_estimators=self.n_estimators)
        self.clf3 = RandomForestClassifier(n_estimators=self.n_estimators)
        self.clf4 = RandomForestClassifier(n_estimators=self.n_estimators)
        #self.clf5 = RandomForestRegressor(n_estimators=self.n_estimators)
        #self.clf6 = RandomForestClassifier(n_estimators=self.n_estimators)

    def create_traces(self, event_df, trace_id='trace_id'):
        ll=[]
        trace_ids=[]
        cols=list(event_df)
        cols.remove(trace_id)
        for n, g in event_df.groupby(trace_id): #progress_bar()
            l=[]
            for c in cols:
                l.append(list(g[c]))
            ll.append(l)
            trace_ids.append(n)
        df=pd.DataFrame(ll,columns=cols)
        df["trace_id"] = trace_ids
        return df

    def drop_short_cases(self, traces, col, index=2):
        return traces[traces[col].map(len) > (index+1)].reset_index(drop=True)

    def build_windows(self, traces, ws, col, index=2):
        val = traces[col].values
        X = []

        for i in range(len(traces)):
            for j in range(len(val[i])):
                if j+ws >= len(val[i]) or j < ws-1: #j+ws+1, ws-1
                    continue
                else:
                    windows = []
                    windows.append(traces["trace_id"][i]) # trace_id
                    windows.append(val[i][j:ws+j])        # windows
                    windows.append(val[i][j])             # a1
                    windows.append(val[i][j+1])           # a2
                    windows.append(val[i][-1])            # last activity/resource = outcome
                    windows.append(val[i][j+ws])          # desired prediction = preds
                    X.append(windows)

        new_df = pd.DataFrame(X, columns=["trace_id", "windows", "a1", "a2", "outcome", "preds"]) #, "duration"
        return new_df

    def train(self, data):
        train_traces = self.create_traces(data)
        short_train = self.drop_short_cases(train_traces, "activity")
        window_df1 = self.build_windows(short_train, 2, "activity")
        window_df2 = self.build_windows(short_train, 2, "resource")

        train1 = window_df1[["a1", "a2"]]   # sliding window next step training
        target1 = window_df1["preds"]
        self.clf1.fit(train1, target1)

        train2 = window_df2[["a1", "a2"]]   # sliding window next resource training
        target2 = window_df2["preds"]
        self.clf2.fit(train2, target2)

        target3 = window_df1["outcome"]     # sliding window outcome training
        self.clf3.fit(train1, target3)

        target4 = window_df2["outcome"]     # sliding window last resource training
        self.clf4.fit(train2, target4)

        return self.clf1, self.clf2, self.clf3, self.clf4

    def predict(self, test, index=2):
        test_traces = self.create_traces(test)
        short_test = self.drop_short_cases(test_traces, "activity")
        test_window1 = self.build_windows(short_test, 2, "activity")
        test_window2 = self.build_windows(short_test, 2, "resource")

        X1 = test_window1[["trace_id", "a1", "a2"]].groupby("trace_id") # nsp
        X2 = test_window2[["trace_id", "a1", "a2"]].groupby("trace_id") # nrp

        preds1, preds2, preds3, preds4 = [], [], [], []
        for i,j in X1:
            z = j[["a1", "a2"]]
            preds1.append(self.clf1.predict(z)) # nsp
            preds3.append(self.clf3.predict(z)) # op
        preds1 = pd.Series(preds1)
        preds3 = pd.Series(preds3)

        for k,l in X2:
            p = l[["a1", "a2"]]
            preds2.append(self.clf2.predict(p)) # nrp
            preds4.append(self.clf4.predict(p)) # lrp
        preds2 = pd.Series(preds2)
        preds4 = pd.Series(preds4)

        self.cases = pd.Series(X1.apply(list).index).to_frame() # get trace_ids
        self.cases["nsp"] = preds1
        self.cases["nrp"] = preds2
        self.cases["op"]  = preds3
        self.cases["lrp"] = preds4
        self.cases = self.cases.set_index("trace_id")
        return super().predict(test,index)

    def name(self): return self.__class__.__name__

    def next_step_prediction(self, test, index=2): return self.cases["nsp"]

    def next_resource_prediction(self, test, index=2): return self.cases["nrp"]

    def last_resource_prediction(self, test, index=2): return self.cases["lrp"]

    def outcome_prediction(self, test, index=2): return self.cases["op"]

    def duration_to_next_event_prediction(self, test, index=2): return pd.Series(dtype=float)
    def duration_to_end_prediction(self, test, index=2):  return pd.Series(dtype=float)
    def activity_suffix_prediction(self, test, index=2):  return pd.Series(dtype=float)
    def resource_suffix_prediction(self, test, index=2):  return pd.Series(dtype=float)

class SimpleRandomForest_PPM(PPModel):
    def __init__(self):
        self.n_estimators = 10 # tune min_sample_split and max_depth???
        self.clf1 = RandomForestClassifier(n_estimators=self.n_estimators)
        self.clf2 = RandomForestClassifier(n_estimators=self.n_estimators)
        self.clf3 = RandomForestClassifier(n_estimators=self.n_estimators)
        self.clf4 = RandomForestClassifier(n_estimators=self.n_estimators)
        self.clf5 = RandomForestRegressor(n_estimators=self.n_estimators)
        self.clf6 = RandomForestRegressor(n_estimators=self.n_estimators)
        #self.clf7 = RandomForestClassifier(n_estimators=self.n_estimators)
        #self.clf8 = RandomForestClassifier(n_estimators=self.n_estimators)

    def predict_results(self, index, clf, col):
        tmp = self.cases.apply(lambda x: x[col].values[index:-1])
        tmp = tmp[tmp.apply(lambda x: len(x)>0)]
        res = tmp.apply(lambda x: clf.predict(x))
        return res

    def train(self, data):
        train1 = data[["activity"]]                        # naives next activity training
        target1 = data["next activity"]
        self.clf1.fit(train1, target1)

        train2 = data[["resource", "activity"]]            # naives next resource training
        target2 = data["next resource"]
        self.clf2.fit(train2, target2)

        target3 = data["last resource"]                    # naives last resource training
        self.clf3.fit(train2, target3)

        target4 = data["outcome"]                          # naives outcome training
        self.clf4.fit(train1, target4)

        train3 = data[["activity", "relative timestamp"]]
        #target5 = data["next relative timestamp"]           # naives duration to next event training
        #self.clf5.fit(train3, target5)

        target5 = data["relative timestamp"]
        self.clf5.fit(train1, target5)

        target6 = data["duration to end"]                  # naives duration to outcome training
        self.clf6.fit(train3, target6)

        #target7 = data["activity suffix"]                  # naives activity suffix training
        #self.clf7.fit(train1, target7)

        #target8 = data["resource suffix"]                  # naives resource suffix training
        #self.clf8.fit(train2, target8)
        return self.clf1, self.clf2, self.clf3, self.clf4, self.clf5, self.clf6 #, self.clf7, self.clf8

    def predict(self, test, index=2):
        self.cases = test.groupby("trace_id")
        return super().predict(test,index)

    def name(self):
        return self.__class__.__name__

    def next_step_prediction(self, test, index=2):
        return self.predict_results(index, self.clf1, ["activity"])

    def next_resource_prediction(self, test, index=2):
        return self.predict_results(index, self.clf2, ["resource", "activity"])

    def last_resource_prediction(self, test, index=2):
        return self.predict_results(index, self.clf3, ["resource", "activity"])

    def outcome_prediction(self, test, index=2):
        return self.predict_results(index, self.clf4, ["activity"])

    def duration_to_next_event_prediction(self, test, index=2):
        return self.predict_results(index, self.clf5, ["activity"]) #, "relative timestamp"])

    def duration_to_end_prediction(self, test, index=2):
        return self.predict_results(index, self.clf6, ["activity", "relative timestamp"])

    def activity_suffix_prediction(self, test, index=2):  return pd.Series(dtype=float)
    def resource_suffix_prediction(self, test, index=2):  return pd.Series(dtype=float)