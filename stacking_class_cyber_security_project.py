import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,f1_score, precision_score, recall_score, accuracy_score

class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.k = 3

    def load_data(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test =  x_test
        self.y_train = y_train
        self.y_test = y_test

    def StackingClassifier(self):
        # Define weak learners
        weak_learners = [('SVM', SVC(kernel='rbf', C=100)), ('rf', RandomForestClassifier(n_estimators=100)), ('kNN', KNeighborsClassifier(n_neighbors=2))]

        # Finaler learner or meta model
        final_learner = LogisticRegression()

        train_meta_model = None
        test_meta_model = None

        # Start stacking
        for clf_id, clf in weak_learners:
            print("Classifier ID: ", clf_id)
            # Predictions for each classifier based on k-fold
            predictions_clf = self.k_fold_cross_validation(clf)

            # Predictions for test set for each classifier based on train of level 0
            test_predictions_clf = self.train_level_0(clf)

            # Stack predictions which will form
            # the inputa data for the data model
            if isinstance(train_meta_model, np.ndarray):
                train_meta_model = np.vstack((train_meta_model, predictions_clf))
            else:
                train_meta_model = predictions_clf

            # Stack predictions from test set
            # which will form test data for meta model
            if isinstance(test_meta_model, np.ndarray):
                test_meta_model = np.vstack((test_meta_model, test_predictions_clf))
            else:
                test_meta_model = test_predictions_clf

        # Transpose train_meta_model
        train_meta_model = train_meta_model.T

        # Transpose test_meta_model
        test_meta_model = test_meta_model.T

        # Training level 1
        self.train_level_1(final_learner, train_meta_model, test_meta_model)

    def k_fold_cross_validation(self, clf):
        print("k-fold cross validation")

        predictions_clf = None

        # Number of samples per fold
        batch_size = int(len(self.x_train) / self.k)

        # Stars k-fold cross validation
        for fold in range(self.k):
            print("fold number: ", fold)
            # Settings for each batch_size
            if fold == (self.k - 1):
                test = self.x_train[(batch_size * fold):, :]
                batch_start = batch_size * fold
                batch_finish = self.x_train.shape[0]
            else:
                test = self.x_train[(batch_size * fold): (batch_size * (fold + 1)), :]
                batch_start = batch_size * fold
                batch_finish = batch_size * (fold + 1)

            # test & training samples for each fold iteration
            fold_x_test = self.x_train[batch_start:batch_finish, :]
            fold_x_train = self.x_train[[index for index in range(self.x_train.shape[0]) if
                                         index not in range(batch_start, batch_finish)], :]

            # test & training targets for each fold iteration
            fold_y_test = self.y_train[batch_start:batch_finish]
            fold_y_train = self.y_train[
                [index for index in range(self.x_train.shape[0]) if index not in range(batch_start, batch_finish)]]

            # Fit current classifier
            clf.fit(fold_x_train, fold_y_train)
            fold_y_pred = clf.predict(fold_x_test)

            # Store predictions for each fold_x_test
            if isinstance(predictions_clf, np.ndarray):
                predictions_clf = np.concatenate((predictions_clf, fold_y_pred))
            else:
                predictions_clf = fold_y_pred

        return predictions_clf

    def train_level_0(self, clf):
        print("train level-0")
        # Train in full real training set
        clf.fit(self.x_train, self.y_train)
        # Get predictions from full real test set
        y_pred = clf.predict(self.x_test)

        return y_pred

    def train_level_1(self, final_learner, train_meta_model, test_meta_model):
        print("train level-1")
        # Train is carried out with final learner or meta model
        final_learner.fit(train_meta_model, self.y_train)
        # Getting train and test accuracies from meta_model
        print(f"Train accuracy: {final_learner.score(train_meta_model, self.y_train)}")
        print(f"Test accuracy: {final_learner.score(test_meta_model, self.y_test)}")

        predictions = final_learner.predict(test_meta_model)

        print('Accuracy Stacking: ', accuracy_score(predictions, self.y_test))
        print('Confusion matrix Stacking: ', confusion_matrix(predictions, self.y_test))
        print('Classification report Stacking: ', classification_report(predictions, self.y_test))
        print('F1-score Stacking: ', f1_score(predictions, self.y_test, average='weighted'))
        print('Precision score Stacking: ', precision_score(predictions, self.y_test, average='weighted'))
        print('Recall score Stacking: ', recall_score(predictions, self.y_test, average='weighted'))

if __name__ == "__main__":
    ensemble = Ensemble()

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

    train.replace(to_replace='normal', value=0, inplace = True)
    train.replace(to_replace='anomaly', value=1, inplace=True)

    test.replace(to_replace='normal', value=0, inplace = True)
    test.replace(to_replace='anomaly', value=1, inplace=True)

    X_train = train.loc[:,
                 ['Src Bytes', 'Dst Bytes', 'Duration', 'Dst Host Srv Count', 'Count', 'Dst Host Count', 'Service',
                  'Flag', 'Dst Host Srv Serror Rate',
                  'Srv Serror Rate', 'Serror Rate', 'Dst Host Serror Rate', 'Logged In', 'Num Root', 'Num Compromised',
                  'Dst Host Same Srv Rate',
                  'Same Srv Rate', 'Srv Rerror Rate', 'Rerror Rate', 'Dst Host Srv Rerror Rate', 'Dst Host Rerror Rate',
                  'Score']]
    X_test = test.loc[:,
                ['Src Bytes', 'Dst Bytes', 'Duration', 'Dst Host Srv Count', 'Count', 'Dst Host Count', 'Service',
                 'Flag', 'Dst Host Srv Serror Rate',
                 'Srv Serror Rate', 'Serror Rate', 'Dst Host Serror Rate', 'Logged In', 'Num Root', 'Num Compromised',
                 'Dst Host Same Srv Rate',
                 'Same Srv Rate', 'Srv Rerror Rate', 'Rerror Rate', 'Dst Host Srv Rerror Rate', 'Dst Host Rerror Rate',
                 'Score']]

    Y_train = train['Label']
    Y_test = test['Label']

    X_train_array = X_train.to_numpy()
    X_test_array = X_test.to_numpy()

    ensemble.load_data(x_train = X_train_array, x_test = X_test_array, y_train = Y_train, y_test = Y_test)
    ensemble.StackingClassifier()