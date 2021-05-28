from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
import numpy as np
import warnings
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

warnings.simplefilter('ignore')

RANDOM_SEED = 42
columns_list = ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Land', 'Wrong Fragment', 'Urgent', 'Hot', 'Num Failed Logins', 'Logged In', 'Num Compromised', 'Root Shell', 'Su Attempted', 'Num Root', 'Num File Creations', 'Num Shells', 'Num Access Files', 'Num Outbound Cmds', 'Is Hot Logins', 'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Same Src Port Rate', 'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Label', 'Score']
nsl_kdd_train_data =  pd.read_csv('KDDTrain+.txt', sep=",", names=columns_list)

# Test data
columns_list = ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Land', 'Wrong Fragment', 'Urgent', 'Hot', 'Num Failed Logins', 'Logged In', 'Num Compromised', 'Root Shell', 'Su Attempted', 'Num Root', 'Num File Creations', 'Num Shells', 'Num Access Files', 'Num Outbound Cmds', 'Is Hot Logins', 'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Same Src Port Rate', 'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Label', 'Score']
nsl_kdd_test_data =  pd.read_csv('KDDTest+.txt', sep=",", names=columns_list)

# nsl_kdd_train_data['split'] ='train'
# nsl_kdd_test_data['split']= 'test'

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

# train = nsl_kdd_combined_dataset[nsl_kdd_combined_dataset.split == "train"]
# test = nsl_kdd_combined_dataset[nsl_kdd_combined_dataset.split == "test"]
# train = train.drop("split", axis=1)
# test = test.drop("split", axis=1)



X = nsl_kdd_combined_dataset.loc[:, [ 'Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Wrong Fragment', 'Logged In', 'Num Compromised', 'Num Root', 'Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Score']]

y = nsl_kdd_combined_dataset['Label']
# print(X_train, y_train)

scaler = MinMaxScaler()
# X = pd.DataFrame(scaler.fit_transform(X))

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = GaussianNB()
lr = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)

# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr,
                            random_state=RANDOM_SEED)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Support Vector',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=5, scoring='accuracy')
    print(scores, clf, label)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))