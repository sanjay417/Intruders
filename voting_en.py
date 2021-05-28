# make a prediction with voting ensemble

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,classification_report
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Train data
columns_list = ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Land', 'Wrong Fragment',
                'Urgent', 'Hot', 'Num Failed Logins', 'Logged In', 'Num Compromised', 'Root Shell', 'Su Attempted',
                'Num Root', 'Num File Creations', 'Num Shells', 'Num Access Files', 'Num Outbound Cmds',
                'Is Hot Logins', 'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate', 'Srv Serror Rate',
                'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate',
                'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate',
                'Dst Host Same Src Port Rate', 'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate',
                'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Label', 'Score']
nsl_kdd_train_data = pd.read_csv('KDDTrain+.txt', sep=",", names=columns_list)

# Test data
columns_list = ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Land', 'Wrong Fragment',
                'Urgent', 'Hot', 'Num Failed Logins', 'Logged In', 'Num Compromised', 'Root Shell', 'Su Attempted',
                'Num Root', 'Num File Creations', 'Num Shells', 'Num Access Files', 'Num Outbound Cmds',
                'Is Hot Logins', 'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate', 'Srv Serror Rate',
                'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate',
                'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate',
                'Dst Host Same Src Port Rate', 'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate',
                'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Label', 'Score']
nsl_kdd_test_data = pd.read_csv('KDDTest+.txt', sep=",", names=columns_list)

nsl_kdd_train_data['split'] = 'train'
nsl_kdd_test_data['split'] = 'test'

# Concat Train and Test set
nsl_kdd_combined_data = pd.concat([nsl_kdd_train_data, nsl_kdd_test_data])

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


X_train= train.loc[:,
            ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Wrong Fragment', 'Logged In',
             'Num Compromised', 'Num Root', 'Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate',
             'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count',
             'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate',
             'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Score']]
X_test= test.loc[:,
            ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Wrong Fragment', 'Logged In',
             'Num Compromised', 'Num Root', 'Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate',
             'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count',
             'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate',
             'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Score']]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train=train['Label']
y_test= test['Label']


clf_list = [('rf', RandomForestClassifier(n_estimators=100)), ('svc', SVC(kernel='rbf', C=100)),('knn', KNeighborsClassifier(n_neighbors = 2))]

for model_tuple in clf_list:
    model = model_tuple[1]
    if 'random_state' in model.get_params().keys():
        model.set_params(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred)
    acc = accuracy_score(y_pred, y_test)
    print(f"{model_tuple[0]}'s accuracy: {acc:.2f}")


voting_clf = VotingClassifier(clf_list, voting='hard')
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print(f"Voting Classifier's accuracy: {accuracy_score(y_pred, y_test):.2f}")