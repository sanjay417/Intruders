import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Train data
columns_list = ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Land', 'Wrong Fragment', 'Urgent', 'Hot', 'Num Failed Logins', 'Logged In', 'Num Compromised', 'Root Shell', 'Su Attempted', 'Num Root', 'Num File Creations', 'Num Shells', 'Num Access Files', 'Num Outbound Cmds', 'Is Hot Logins', 'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Same Src Port Rate', 'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Label', 'Score']
nsl_kdd_train_data =  pd.read_csv('KDDTrain+.txt', sep=",", names=columns_list)

# Test data
columns_list = ['Duration', 'Protocol Type', 'Service', 'Flag', 'Src Bytes', 'Dst Bytes', 'Land', 'Wrong Fragment', 'Urgent', 'Hot', 'Num Failed Logins', 'Logged In', 'Num Compromised', 'Root Shell', 'Su Attempted', 'Num Root', 'Num File Creations', 'Num Shells', 'Num Access Files', 'Num Outbound Cmds', 'Is Hot Logins', 'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate', 'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate', 'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count', 'Dst Host Srv Count', 'Dst Host Same Srv Rate', 'Dst Host Diff Srv Rate', 'Dst Host Same Src Port Rate', 'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate', 'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate', 'Dst Host Srv Rerror Rate', 'Label', 'Score']
nsl_kdd_test_data =  pd.read_csv('KDDTest+.txt', sep=",", names=columns_list)

def traindata_processing(nsl_kdd_train_data):
    nsl_kdd_train_data_labeled = nsl_kdd_train_data.copy()
    # grouping anomalies
    nsl_kdd_train_data_labeled.replace(to_replace=['neptune', 'warezclient', 'ipsweep', 'portsweep',
                                                   'teardrop', 'nmap', 'satan', 'smurf', 'pod', 'back',
                                                   'guess_passwd', 'ftp_write', 'multihop', 'rootkit',
                                                   'buffer_overflow', 'imap', 'warezmaster', 'phf', 'land',
                                                   'loadmodule', 'spy', 'perl'], value='anomaly', inplace=True)

    nsl_kdd_train_data_labeled_copy = nsl_kdd_train_data_labeled.copy()

    # SVM cannot process categorical data so convert Service, Protocol Type, Flag into numerical data
    # Service
    nsl_kdd_train_data_labeled_service_copy = nsl_kdd_train_data_labeled_copy.copy()

    nsl_kdd_train_data_labeled_service_copy.reset_index(drop=True, inplace=True)
    service_dummies = pd.get_dummies(nsl_kdd_train_data_labeled_service_copy['Service'], prefix='service')

    # Protocol Type
    nsl_kdd_train_data_labeled_protocol_copy = nsl_kdd_train_data_labeled_copy.copy()
    nsl_kdd_train_data_labeled_protocol_copy.reset_index(drop=True, inplace=True)

    protocol_dummies = pd.get_dummies(nsl_kdd_train_data_labeled_protocol_copy['Protocol Type'], prefix='protocol')

    # Flag
    nsl_kdd_train_data_labeled_flag_copy = nsl_kdd_train_data_labeled_copy.copy()
    nsl_kdd_train_data_labeled_flag_copy.reset_index(drop=True, inplace=True)

    flag_dummies = pd.get_dummies(nsl_kdd_train_data_labeled_flag_copy['Flag'], prefix='flag')

    nsl_kdd_train_data_labeled_join_copy = nsl_kdd_train_data_labeled_copy.copy()
    nsl_kdd_train_data_labeled_join_copy.reset_index(drop=True, inplace=True)

    # Join the dummy columns with the train dataset
    nsl_kdd_train_data_labeled_join_copy = pd.concat(
        [nsl_kdd_train_data_labeled_join_copy, protocol_dummies, service_dummies, flag_dummies], axis=1)

    # Drop redundant columns
    nsl_kdd_train_data_labeled_final_copy = nsl_kdd_train_data_labeled_join_copy.drop(
        ['Protocol Type', 'Service', 'Flag', 'service_red_i', 'service_urh_i', 'service_http_8001', 'service_http_2784',
         'service_harvest', 'service_aol', 'Score'], axis=1)

    nsl_kdd_train_data_preprocessed = nsl_kdd_train_data_labeled_final_copy.copy()

    return nsl_kdd_train_data_preprocessed

def testdata_processing(nsl_kdd_test_data):
    nsl_kdd_test_data_labeled = nsl_kdd_test_data.copy()
    # grouping anomalies
    nsl_kdd_test_data_labeled.replace(to_replace=['neptune', 'saint', 'mscan', 'guess_passwd', 'smurf',
           'apache2', 'satan', 'buffer_overflow', 'back', 'warezmaster',
           'snmpgetattack', 'processtable', 'pod', 'httptunnel', 'nmap', 'ps',
           'snmpguess', 'ipsweep', 'mailbomb', 'portsweep', 'multihop',
           'named', 'sendmail', 'loadmodule', 'xterm', 'worm', 'teardrop',
           'rootkit', 'xlock', 'perl', 'land', 'xsnoop', 'sqlattack',
           'ftp_write', 'imap', 'udpstorm', 'phf'], value='anomaly', inplace=True)


    nsl_kdd_test_data_labeled_copy = nsl_kdd_test_data_labeled.copy()
    # Service
    nsl_kdd_test_data_labeled_service_copy = nsl_kdd_test_data_labeled_copy.copy()
    nsl_kdd_test_data_labeled_service_copy.reset_index(drop=True, inplace=True)

    service_dummies = pd.get_dummies(nsl_kdd_test_data_labeled_service_copy['Service'], prefix='service')

    # Protocol Type
    nsl_kdd_test_data_labeled_protocol_copy = nsl_kdd_test_data_labeled_copy.copy()
    nsl_kdd_test_data_labeled_protocol_copy.reset_index(drop=True, inplace=True)

    protocol_dummies = pd.get_dummies(nsl_kdd_test_data_labeled_protocol_copy['Protocol Type'], prefix='protocol')

    # Flag
    nsl_kdd_test_data_labeled_flag_copy = nsl_kdd_test_data_labeled_copy.copy()
    nsl_kdd_test_data_labeled_flag_copy.reset_index(drop=True, inplace=True)

    flag_dummies = pd.get_dummies(nsl_kdd_test_data_labeled_flag_copy['Flag'], prefix='flag')

    nsl_kdd_test_data_labeled_join_copy = nsl_kdd_test_data_labeled_copy.copy()
    nsl_kdd_test_data_labeled_join_copy.reset_index(drop=True, inplace=True)

    # Join the dummy columns with the test dataset
    nsl_kdd_test_data_labeled_join_copy = pd.concat([nsl_kdd_test_data_labeled_join_copy, protocol_dummies,service_dummies,flag_dummies], axis=1)

    # Drop redundant columns
    nsl_kdd_test_data_labeled_final_copy = nsl_kdd_test_data_labeled_join_copy.drop(['Protocol Type','Service','Flag','Score'], axis=1)

    nsl_kdd_test_data_preprocessed = nsl_kdd_test_data_labeled_final_copy.copy()

    return nsl_kdd_test_data_preprocessed

nsl_kdd_train_data_preprocessed = traindata_processing(nsl_kdd_train_data)
nsl_kdd_test_data_preprocessed = testdata_processing (nsl_kdd_test_data)

# Train/test split
X_train = nsl_kdd_train_data_preprocessed.drop('Label', axis=1)
y_train = nsl_kdd_train_data_preprocessed['Label']

X_test = nsl_kdd_test_data_preprocessed.drop('Label', axis=1)
y_test = nsl_kdd_test_data_preprocessed['Label']

# Standardizing the data
scaler = StandardScaler()
scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train))
scaled_X_test = pd.DataFrame(scaler.fit_transform(X_test))

# Support Vector Machine
model = SVC(kernel='rbf')
model.fit(scaled_X_train, y_train)

predictions = model.predict(scaled_X_test)
expected = y_test

# Evaluation results
print('accuracy ', metrics.accuracy_score(expected, predictions))
print('confusion matrix ', confusion_matrix(expected, predictions))
print('classification report ', classification_report(expected, predictions))
