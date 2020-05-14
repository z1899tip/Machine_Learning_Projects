import pandas as pd
from zipfile import ZipFile
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')


# 1. Load/Extract the data

start = time.perf_counter()

tar_file = '/home/savoroso/Downloads'  # change with your target address
ex_path = os.path.join(tar_file,'titanic.zip')
tar_path = os.path.join(tar_file,'data_file')
if not os.path.isdir(tar_path):
	os.makedirs(tar_path)

with ZipFile(ex_path,'r') as data_unzip:
	data_unzip.extractall(tar_path)

w_file_train = os.path.join(tar_path,'train.csv')
w_file_test = os.path.join(tar_path,'test.csv')

df_train = pd.read_csv(w_file_train)	
df_test = pd.read_csv(w_file_test)


#2 check and manage missing values

def missing_values(df):
	df_mv = df.isnull().sum()/df.isnull().count() * 100
	df_mv = df_mv[df_mv>0]

	sns.barplot(df_mv.index,df_mv)
	plt.xlabel('feature')
	plt.ylabel('%')
	plt.show()
	return df_mv

# drop cabin due to high ratio of missing values
df_test.drop('Cabin',axis = 1,inplace=True)
df_train.drop('Cabin',axis = 1,inplace=True)


def fill_missing_values(df):
	for col_name in df.columns.values.tolist():
		if df[col_name].dtype in (np.int64,np.float64):	  
			col_name_mean = df[col_name].mean()
			df[col_name].fillna(col_name_mean,inplace=True)

		elif df[col_name].dtype == object:
			df_mode = df[col_name].mode()[0]
			df[col_name].fillna(df_mode,inplace=True)

	return df


#3. Feature Engineering (check for outliers/transform categorical to numerical values/ Binning)


d_train = df_train.copy()
d_test = df_test.copy()

def simplify_Fam_size(dataset):
	dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+1 
	return dataset

def simplify_fare(dataset):
	dataset['Fare_Bin'] = pd.cut(dataset['Fare'],bins=[0,8,14,31,513],labels=['Low Fare','Medium Fare','Average Fare','High Fare'], include_lowest=True)
	return dataset

def simplify_age(dataset):
	dataset['Age_Bin'] = pd.cut(dataset['Age'],bins=(0,10,20,40,100),labels = ['Child','Teenager','Adult','Old'],include_lowest=True)
	return dataset

def simplify_name(dataset):
	dataset['name_pref'] = dataset['Name'].apply(lambda x: x.split(' ')[1])
	dataset['name_pref'] = dataset['name_pref'].replace('Ms.','Miss.')
	dataset['name_pref'] = dataset['name_pref'].replace(['Don.','Master.'],'Mr.')
	dataset.loc[~dataset['name_pref'].isin(['Mr.','Miss.','Mrs.']),'name_pref'] = 'Rare'
	return dataset

def simplify_drop(dataset):
	columns_drop = ['Age','Fare','SibSp','PassengerId','Parch','Name','Ticket']
	# columns_drop = ['Age','Fare','Name','PassengerId','Ticket']	
	dataset.drop(columns_drop,axis=1,inplace=True)
	return dataset

def transform(df):
	df = fill_missing_values(df)
	df = simplify_Fam_size(df)
	df = simplify_age(df)
	df = simplify_fare(df)
	df = simplify_name(df)
	df = simplify_drop(df)
	return df


d_train.drop('PassengerId',axis=1)
df_train_trans = transform(d_train)
df_test_trans = transform(d_test)

# print(df_train_trans.columns.values.tolist())
# print(df_train_trans.isnull().sum())
# print(df_test_trans.isnull().sum())

# check for columns with categorical and convert to numerical
# for col in df_train_trans.columns.values.tolist():
# 	print(col,":",df_train_trans[col].dtype)

# Object
# sex/Ticket/Embarked/name_pref

def trans_cat_num(df):
	col_list = []
	for col in df.columns.values.tolist():
		if df[col].dtype.name in ('object','category'):
			col_list.append(col)
	df = pd.get_dummies(df,columns=col_list,prefix=col_list)
	return df

df_transformed = trans_cat_num(df_train_trans)

def corr_map(df):
	sns.heatmap(df.corr(),annot = True, cmap='jet',linewidths=0.1)
	fig = plt.gcf()
	fig.set_size_inches(20,12)
	plt.show()

# corr_map(df_transformed)


# 4. split the data to train and test

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import recall_score,precision_score,precision_recall_curve


#seperate the label and its attributes
df_attrib = df_transformed.drop('Survived',axis=1)
df_label = df_transformed['Survived']

X_train,X_test,y_train,y_test = train_test_split(df_attrib,df_label,test_size = 0.3,random_state=42)

# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


#5. Select model (I will choose model with higher accuracy)
#Classification: Random Forest / Decision Tree / KNN / SVM / Logistic Regression / Naive Bayes Classifier / Linear Discriminant Analysis / Ada Boost Classifier / Gradient Boosting Classifier


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


## 6. Select Best model
##6.1  Random Forest Classifier

# rfc = RandomForestClassifier()

# rfc.fit(X_train,y_train)
# prediction_value = rfc.predict(X_test)
# print('The accuracy of Random Forest is:',round(accuracy_score(prediction_value,y_test)*100,2))

# kfl = KFold(n_splits=10,random_state=42) # split the data in to 10parts
# res = cross_val_score(rfc,df_attrib,df_label,cv=10,scoring='accuracy')
# print('the cross validated score is:',round(res.mean()*100,2))


# # confusion matrix
# y_pred = cross_val_predict(rfc,df_attrib,df_label,cv=10)

# # sns.heatmap(confusion_matrix(df_label,y_pred),annot=True,fmt='3.0f',cmap='jet')

# print(confusion_matrix(df_label,y_pred))
# print('recall:',round(recall_score(df_label,y_pred)*100,2))
# print('precision:',round(precision_score(df_label,y_pred)*100,2))

##recall_score ---> tp/tp+fn 
## precision_score --> tp/tp+fp


kfl = KFold(n_splits=10,random_state=42) # will be used in Gridsearch; split data into 10parts


def scoring_func(model_name,model,pred_model,d_attrib,d_label,ytest):
	accscore = accuracy_score(pred_model,ytest)
	# print(f'{model_name} Accuracy:',round(accscore*100,2))
	cvs_res = cross_val_score(model,d_attrib,d_label,scoring='accuracy',cv=10)
	# print(f'{model_name} Cross Validation Score:',round(cvs_res.mean()*100,2))

	y_pred = cross_val_predict(model,d_attrib,d_label,cv=10)
	# print(confusion_matrix(d_label,y_pred))

	pscore = precision_score(d_label,y_pred)
	rscore = recall_score(d_label,y_pred)

	# print(f'{model_name} Precision:',round(pscore*100,2))
	# print(f'{model_name} Recall:',round(rscore*100,2))

	return cvs_res,accscore,pscore,rscore


##########################RandomForestClassifier#################################

print('1','+'*30)

rfc = RandomForestClassifier()
# rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=None, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=300,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False
# 	)

rfc.fit(X_train,y_train)
pred_model_rfc = rfc.predict(X_test)

cv_score_rfc,acc_score_rfc,prec_score_rfc,rec_score_rfc = scoring_func('RFC',rfc,pred_model_rfc,df_attrib,df_label,y_test)

# # importances = rfc.feature_importances_

# # plt.bar(X_train.columns,importances)
# # plt.show()

# # df_importance = pd.DataFrame(importances,index=X_train.columns)
# # print(df_importance.sort_values(by =0 , inplace=False))

# # temp_store = []
# # for k,i in enumerate(importances):
# # 	# print(k,i)
# # 	temp_store.append((X_train.columns[k],':',round(i*100,2)))

# # print(temp_store.sort())

# # std = np.std([tree.feature_importances_ for tree in rfc.estimators_],axis=0)
# # indices = np.argsort(importances)[::-1]


print('2','+'*30)

######################################### Decision Tree###########################
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
pred_model_dtc = dtc.predict(X_test)

cv_score_dtc,acc_score_dtc,prec_score_dtc,rec_score_dtc = scoring_func('DTC',dtc,pred_model_dtc,df_attrib,df_label,y_test)


print('3','+'*30)

##############################KNN#################################################
knc = KNeighborsClassifier()

knc.fit(X_train,y_train)
pred_model_knc = knc.predict(X_test)

cv_score_knc,acc_score_knc,prec_score_knc,rec_score_knc = scoring_func('KNC',knc,pred_model_knc,df_attrib,df_label,y_test)

print('4','+'*30)

################################SVM################################################

from sklearn.svm import SVC

svmc = SVC()
# svmc = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False
# 	)

svmc.fit(X_train,y_train)
pred_model_svmc = svmc.predict(X_test)


# dx = pd.DataFrame(df_attrib.iloc[101])
# pred_model_svmc = svmc.predict(dx.T)
# # print(df_label[])
# # print(len(pred_model_svmc))
# # print(len(X_test))

# target_label = np.unique(df_label)
# preds = target_label[pred_model_svmc]
# print(preds)
# print('ans:',df_label.iloc[101])

cv_score_svmc,acc_score_svmc,prec_score_svmc,rec_score_svmc = scoring_func('SVMC',svmc,pred_model_svmc,df_attrib,df_label,y_test)


print('5','+'*30)

#####################################logistic Regression#######################################

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)
pred_model_lr = lr.predict(X_test)
cv_score_lr,acc_score_lr,prec_score_lr,rec_score_lr = scoring_func('LR',lr,pred_model_lr,df_attrib,df_label,y_test)



print('6','+'*30)

########################################gausian naive bayes######################################

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()


gnb.fit(X_train,y_train)
pred_model_gnb = gnb.predict(X_test)

cv_score_gnb,acc_score_gnb,prec_score_gnb,rec_score_gnb = scoring_func('GNB',gnb,pred_model_gnb,df_attrib,df_label,y_test)


print('7','+'*30)

##################################ADAboost Classifier#############################################

from sklearn.ensemble import AdaBoostClassifier
adab = AdaBoostClassifier()


adab.fit(X_train,y_train)
pred_model_adab = adab.predict(X_test)

cv_score_adab,acc_score_adab,prec_score_adab,rec_score_adab = scoring_func('ADAB',adab,pred_model_adab,df_attrib,df_label,y_test)


###################################XGboost Classifier######################################

from xgboost import XGBClassifier
xg = XGBClassifier()

###With Tuning
# xg = XGBClassifier(colsample_bytree=0.7, learning_rate = 0.05, max_depth= 7,min_child_weight= 5, missing = -999,
#  n_estimators = 1000, objective = 'binary:logistic', silent=1, subsample=0.8)


xg.fit(X_train,y_train)
pred_model_xg = xg.predict(X_test)

cv_score_xg,acc_score_xg,prec_score_xg,rec_score_xg = scoring_func('XG',xg,pred_model_xg,df_attrib,df_label,y_test)




print('8','+'*30)

#######################linear discriminant Analysis################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()


lda.fit(X_train,y_train)
pred_model_lda = lda.predict(X_test)

cv_score_lda,acc_score_lda,prec_score_lda,rec_score_lda = scoring_func('LDA',lda,pred_model_lda,df_attrib,df_label,y_test)


print('9','+'*30)

######################Gradient Boosting Classifier###########################

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()


gbc.fit(X_train,y_train)
pred_model_gbc = gbc.predict(X_test)

cv_score_gbc,acc_score_gbc,prec_score_gbc,rec_score_gbc = scoring_func('GBC',gbc,pred_model_gbc,df_attrib,df_label,y_test)



#########################lightGBM##################################

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm.fit(X_train,y_train)
pred_model_lgbm = lgbm.predict(X_test)

cv_score_lgbm,acc_score_lgbm,prec_score_lgbm,rec_score_lgbm = scoring_func('lgbm',lgbm,pred_model_lgbm,df_attrib,df_label,y_test)
# end = time.perf_counter()

# print(f'Done in {end-start} second(s)')






model_cv_score = pd.DataFrame({
	'Model':['Random Forest','Decision Tree','KNN','Naive Bayes','support vector machine'
	,'Logistic Regression','Linear Discriminant Analysis','AdaBoostClassifier','Gradient Decent','XGboost','lighGBM'],
	'Score':[cv_score_rfc.mean(),cv_score_dtc.mean(),cv_score_knc.mean(),cv_score_gnb.mean()
	,cv_score_svmc.mean(),cv_score_lr.mean(),cv_score_lda.mean(),cv_score_adab.mean(),cv_score_gbc.mean(),cv_score_xg.mean(),cv_score_lgbm.mean()]})

model_cv_score.sort_values(by='Score',inplace=True,ascending=False)
print('Model Accuracy Rank Result cv=10')
print(model_cv_score)

# Model Accuracy Rank Result
#                            Model     Score
# 4         support vector machine  0.824627
# 8                Gradient Decent  0.813433
# 9                        XGboost  0.813433
# 10                       lighGBM  0.809701
# 3                    Naive Bayes  0.802239
# 6   Linear Discriminant Analysis  0.802239
# 5            Logistic Regression  0.798507
# 0                  Random Forest  0.791045
# 1                  Decision Tree  0.791045
# 7             AdaBoostClassifier  0.791045
# 2                            KNN  0.787313

# Model Accuracy Rank Result cv=10
#                            Model     Score
# 10                       lighGBM  0.832821
# 4         support vector machine  0.829413
# 2                            KNN  0.826080
# 9                        XGboost  0.823808
# 8                Gradient Decent  0.815968
# 1                  Decision Tree  0.813745
# 0                  Random Forest  0.810424
# 7             AdaBoostClassifier  0.809176
# 5            Logistic Regression  0.808090
# 6   Linear Discriminant Analysis  0.806979
# 3                    Naive Bayes  0.792372





#7.21

###multiprocessing###
import concurrent.futures

def execute_model(model):

	target_model = model()
	target_model.fit(X_train,y_train)
	pred_model = target_model.predict(X_test)
	cv_score,acc_score,prec_score,rec_score = scoring_func('f{model}',target_model,pred_model,df_attrib,df_label,y_test)
	# return f'{model:}',cv_score.mean(),acc_score.mean(),prec_score,rec_score 
	return f'{model:}',cv_score.mean() 

model = [RandomForestClassifier,
KNeighborsClassifier,
DecisionTreeClassifier,
SVC,
LogisticRegression,
GaussianNB,
AdaBoostClassifier,
XGBClassifier,
LinearDiscriminantAnalysis,
GradientBoostingClassifier,
LGBMClassifier]

#####comment out below lines to enable concurrent futures.

# with concurrent.futures.ProcessPoolExecutor() as executor:
# 	results = executor.map(execute_model,model)
# 	for result in results:
# 		print(result)


# end = time.perf_counter()

# print(f'Done in {end-start} second(s)')


# HyperParameter Tuning. (Optimizing Model Default Settings)

from sklearn.model_selection import GridSearchCV


####################rfc tuning##############################
# rfc_t = RandomForestClassifier()
# val = range(100,1000,100)
# param = {"n_estimators":val},
# rfc_gs = GridSearchCV(rfc_t,param_grid = param,cv=5,scoring='accuracy')

# rfc_gs.fit(X_train,y_train)

# print('random forest tuned:',rfc_gs.best_score_)
# print(rfc_gs.best_estimator_)


######################SVC tuning#############################
# svc_t = SVC()

# param = [{'C':[1,10,30,50,100,200,300],'kernel':['rbf','linear','sigmoid','poly'],'gamma':[0.1,0.01,1]}]

# svc_gs = GridSearchCV(svc_t,param_grid=param,cv=kfl,scoring='accuracy')

# svc_gs.fit(X_train,y_train)

# print('svc Score Tuned:',svc_gs.best_score_)
# print(svc_gs.best_estimator_)


##################XGboost Tuning####################


# xg_t = XGBClassifier()
# param = {'objective':['binary:logistic'],
# 'learning_rate':[0.05],
# 'max_depth':range(5,10,1),
# 'min_child_weight':range(1,6,2),
# 'silent':[1],
# 'subsample':[0.8],
# 'colsample_bytree':[0.7],
# 'n_estimators':[1000],
# 'missing':[-999],
# 'seed':[1337]}

# print('Ongoing')


# xg_clf = GridSearchCV(xg_t,param,n_jobs=5,cv=kfl,scoring='roc_auc')

# xg_clf.fit(X_train,y_train)
# print(xg_clf.best_params_,xg_clf.best_score_)


###################LightGBM########################

# lgbm_t = LGBMClassifier(boosting_type='gbdt',  objective='binary', num_boost_round=2000, learning_rate=0.01)

# param_grid = {'num_leaves':[30,127],'min_data_in_leaf':[25,50,75,100,200,300,400]}

# gsearch = GridSearchCV(lgbm_t,param_grid,n_jobs=5,cv=kfl,scoring='roc_auc')
# lgbm = gsearch.fit(X_train,y_train)

# print(lgbm.best_params_,lgbm.best_score_)



# lgbm_t = LGBMClassifier(boosting_type='gbdt', objective='binary', num_boost_round=2000, learning_rate=0.01,min_data_in_leaf=25,num_leaves=30)
# lgbm_t.fit(X_train,y_train)
# pred_model_lgbm_t = lgbm_t.predict(X_test)


# cv_score_lgbm_t,acc_score_lgbm_t,prec_score_lgbm_t,rec_score_lgbm_t = scoring_func('lgbm',lgbm_t,pred_model_lgbm_t,df_attrib,df_label,y_test)

# print('lgbm_tuned:',cv_score_lgbm_t.mean())


# Final Selected Model


df_test_transformed = trans_cat_num(df_test_trans)

### (Test Score: 82.9%, Actual: 77.09%)

# from sklearn.svm import SVC
# model svmc = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False
# 	)


### (Test Score: 82.4%, Actual: 77.09%)
# from xgboost import XGBClassifier
# model = XGBClassifier(colsample_bytree=0.7, learning_rate = 0.05, max_depth= 7,min_child_weight= 5, missing = -999,
#  n_estimators = 10000, objective = 'binary:logistic', silent=1, subsample=0.8)



#### overfitting.. (Test Score: 83.3% - Actual: 77.03%)
model = LGBMClassifier()
model.fit(X_train,y_train)
y_pred_model = model.predict(df_test_transformed)


submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_pred_model})
submission.to_csv('submission',index=False)
