import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix,f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

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

X_train= train.drop('Label', axis=1)
X_test=test.drop('Label', axis=1)
y_train=train['Label']
y_test= test['Label']

# Random Forest
# create the classifier
classifier = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
expected = y_test

# Evaluation results
print('Accuracy RF: ', metrics.accuracy_score(expected, predictions))
print('Confusion matrix RF: ', confusion_matrix(expected, predictions))
print('Classification report RF: ', classification_report(expected, predictions))
print('F1-score RF: ', f1_score(expected, predictions, average='weighted'))
print('Precision score RF: ', precision_score(expected, predictions, average='weighted'))
print('Recall score RF: ', recall_score(expected, predictions, average='weighted'))

# SVM
X_train= train.drop('Label', axis=1)
X_test=test.drop('Label', axis=1)
y_train=train['Label']
y_test= test['Label']


# Standardizing the data
scaler = StandardScaler()
scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train))
scaled_X_test = pd.DataFrame(scaler.fit_transform(X_test))

# Support Vector Machine
model = SVC(kernel='rbf', C=100)
model.fit(scaled_X_train, y_train)

predictions = model.predict(scaled_X_test)
expected = y_test

# Evaluation results
print('Accuracy SVC: ', metrics.accuracy_score(expected, predictions))
print('Confusion matrix SVC: ', confusion_matrix(expected, predictions))
print('Classification report SVC: ', classification_report(expected, predictions))
print('F1-score SVC: ', f1_score(expected, predictions, average='weighted'))
print('Precision score SVC: ', precision_score(expected, predictions, average='weighted'))
print('Recall score SVC: ', recall_score(expected, predictions, average='weighted'))