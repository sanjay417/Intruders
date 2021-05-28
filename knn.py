import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


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

X_train = train.loc[:, [ 'Duration', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Wrong Fragment', 'Count', 'Serror Rate', 'Srv Rerror Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate' ]]
X_test = test.loc[:, [ 'Duration', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Wrong Fragment', 'Count', 'Serror Rate', 'Srv Rerror Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate' ]]
y_train = train['Label']
y_test = test['Label']

# def evaluate_model(model, X, y, k):
#     cv = KFold(n_splits=k, shuffle=True, random_state=1)
#     scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#     print("K -> %d, scores -> %s" %(k, scores))
#     return scores    

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 2)
# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                     reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
classifier.fit(X_train, y_train)
print("KNN -> ",accuracy_score(y_test,classifier.predict(X_test)))

# xgb_predicted = xgb.predict(X_test)

rf = RandomForestClassifier(n_estimators=12, random_state=1, max_depth=7)
rf.fit(X_train,y_train)
rf_predicted = rf.predict(X_test)
print(accuracy_score(y_test, rf_predicted))

# y_pred = classifier.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)

# print('Accuracy SVC: ', accuracy_score(y_test, y_pred))
# print('Confusion matrix SVC: ', confusion_matrix(y_test, y_pred))
# print('Classification report SVC: ', classification_report(y_test, y_pred))
# print('F1-score SVC: ', f1_score(y_test, y_pred, average='weighted'))
# print('Precision score SVC: ', precision_score(y_test, y_pred, average='weighted'))
# print('Recall score SVC: ', recall_score(y_test, y_pred, average='weighted'))