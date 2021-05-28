# make a prediction with a stacking ensemble
from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from numpy import mean
from numpy import std
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold  

# Train data
columns_list = ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Land', 'Wrong Fragment', 'Urgent', 'Hot', 'Num Failed Logins', 'Logged In', 'Num Compromised', 'Root Shell', 'Su Attempted', 'Num Root', 'Num File Creations', 'Num Shells', 'Num Access Files', 'Num Outbound Cmds', 'Is Hot Logins', 'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Same Src Port Rate', 'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Label', 'Score']
nsl_kdd_train_data =  pd.read_csv('KDDTrain+.txt', sep=",", names=columns_list)

# Test data
columns_list = ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Land', 'Wrong Fragment', 'Urgent', 'Hot', 'Num Failed Logins', 'Logged In', 'Num Compromised', 'Root Shell', 'Su Attempted', 'Num Root', 'Num File Creations', 'Num Shells', 'Num Access Files', 'Num Outbound Cmds', 'Is Hot Logins', 'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Same Src Port Rate', 'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Label', 'Score']
nsl_kdd_test_data =  pd.read_csv('KDDTest+.txt', sep=",", names=columns_list)

nsl_kdd_train_data['split'] ='train'
nsl_kdd_test_data['split']= 'test'

# Concat Train and Test set
nsl_kdd_combined_data = pd.concat([nsl_kdd_train_data,nsl_kdd_test_data])

# group label column
nsl_kdd_combined_data_grouped = nsl_kdd_combined_data.copy()

nsl_kdd_combined_data_grouped.replace(to_replace=['neptune', 'warezclient', 'ipsweep', 'portsweep',
       'teardrop', 'nmap', 'satan', 'smurf', 'pod', 'back',
       'guess_passwd', 'ftp_write', 'multihop', 'rootkit',
       'buffer_overflow', 'imap', 'warezmaster', 'phf', 'land',	
       'loadmodule', 'spy', 'perl', 'saint', 'mscan', 'apache2',
       'snmpgetattack', 'processtable', 'httptunnel', 'ps', 'snmpguess',
       'mailbomb', 'named', 'sendmail', 'xterm', 'worm', 'xlock',
       'xsnoop', 'sqlattack', 'udpstorm'], value='anomaly', inplace=True)

# Convert string columns to numerical columns

nsl_kdd_combined_data_enc = nsl_kdd_combined_data_grouped.copy()
cat_columns = ['Protocol Type', 'Service', 'Flag']
encoder = LabelEncoder()
for col in cat_columns:
    nsl_kdd_combined_data_enc[col] = encoder.fit_transform(nsl_kdd_combined_data_enc[col])

nsl_kdd_combined_dataset = nsl_kdd_combined_data_enc.copy()

train = nsl_kdd_combined_dataset[nsl_kdd_combined_dataset.split == "train"]
test = nsl_kdd_combined_dataset[nsl_kdd_combined_dataset.split == "test"]
train = train.drop("split", axis=1)
test = test.drop("split", axis=1)



X_train = train.loc[:, [ 'Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Wrong Fragment', 'Logged In', 'Num Compromised', 'Num Root', 'Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Score']]
# rf_train = train.loc[:, [ 'Duration', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Wrong Fragment', 'Count', 'Serror Rate', 'Srv Rerror Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate' ]]

X_test = test.loc[:, [ 'Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Wrong Fragment', 'Logged In', 'Num Compromised', 'Num Root', 'Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Score']]
# rf_test = test.loc[:, [ 'Duration', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Wrong Fragment', 'Count', 'Serror Rate', 'Srv Rerror Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate' ]]
# X_train= train.drop('Label', axis=1)
# X_test=test.drop('Label', axis=1)
y_train=train['Label']
y_test= test['Label']
# print(X_train, y_train)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.fit_transform(X_test))

# define the base models
def get_stacking():	
	# define the base models
	level0 = list()
	level0.append(('knn', KNeighborsClassifier(n_neighbors=10)))
	level0.append(('cart', RandomForestClassifier(n_estimators=12, random_state=42, max_depth=7)))
	level0.append(('svm', SVC(kernel='rbf', C=100)))
	# level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 =  SVC(kernel='rbf', C=100)
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0,final_estimator=level1, stack_method='predict', cv=5, passthrough=True)
	return model

def get_models():
	models = dict()
	models['knn'] = KNeighborsClassifier(n_neighbors=10)
	models['cart'] = RandomForestClassifier(n_estimators=12, random_state=42, max_depth=7)
	models['svm'] = SVC(kernel='rbf', C=100)
	# models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models

def evaluate_model(model, X, y):
	# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=10)
	print(scores)
	return scores        

# fit the model on all available data
# model.fit(X, y)
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    	
	if name == 'cart':
		print("IN CART")

		#scores = evaluate_model(model, X_train, y_train)
		# results.append(scores)
		# names.append(name)
		# print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores))) 
		model.fit(X_train,y_train)
		y_pred = model.predict(X_test)
		print(accuracy_score(y_test,y_pred))

	elif name == 'svm':
		print("IN SVM")
		
		# scores = evaluate_model(model, X_train, y_train)
		# results.append(scores)
		# names.append(name)	
		# print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores))) 
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		print(accuracy_score(y_test,y_pred))
           
	elif name == 'knn':
		print("In KNN")
		
		# scores = evaluate_model(model, X_train, y_train)
		# results.append(scores)
		# names.append(name)
		# print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		print(accuracy_score(y_test,y_pred))
		

	else:
		print("stacking")
		
		# scores = evaluate_model(model, X_train, y_train)
		# print(scores)
		# results.append(scores)
		# names.append(name)
		# print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))\
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		print(accuracy_score(y_test,y_pred))

# print(model,name)
# scaler = StandardScaler()
# X_test = pd.DataFrame(scaler.fit_transform(X_test))
# y_pred = model.predict(X_test)
# y_test = y_test

# print('Accuracy SVC: ', accuracy_score(y_test, y_pred))
