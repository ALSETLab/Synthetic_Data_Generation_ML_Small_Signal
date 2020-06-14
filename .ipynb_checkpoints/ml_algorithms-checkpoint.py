from sklearn.linear_model import LogisticRegression
import time

def logistic_regression(X_train, X_train_not_red, X_test, Y_train, Y_test):
    # Logistic regression algorithm
    log_reg = LogisticRegression(solver = 'lbfgs', C = 10)

    t_0 = time.time()
    log_reg.fit(X_train, Y_train)
    t_f = time.time() - t_0
    print("Training time (logistic regression prediction): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediciton time

    # Computing predictions on training and testing sets
    predictions_log_reg_train = log_reg.predict(X_train)

    t_0 = time.time() # Measuring overall prediction time
    predictions_log_reg_full_train = log_reg.predict(X_train_not_red)
    predictions_log_reg_test = log_reg.predict(X_test)

    print("Prediction time (logistic regression prediction): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediciton time

    # Computing accuracies
    training_accuracy_log_reg = np.sum(np.equal(Y_train, predictions_log_reg_train))/Y_train.shape[0] * 100
    testing_accuracy_log_reg = np.sum(np.equal(Y_test, predictions_log_reg_test))/Y_test.shape[0] * 100

    # Performance Metrics
    print("Confusion matrix (logistic regression):\n{}".format(confusion_matrix(Y_test, predictions_log_reg_test)))
    print("Accuracy (logistic regression): {:.2f}% (training) - {:.2f}% (testing)".format(training_accuracy_log_reg,
                                                                                 testing_accuracy_log_reg))
    print("Precision (logistic regression): {:.2f}% (training) - {:.2f}% (testing)".format(precision_score(Y_train, 
                                                                                                           predictions_log_reg_train,
                                                                                                           average = 'macro')*100,
                                                                                 precision_score(Y_test, 
                                                                                                 predictions_log_reg_test, 
                                                                                                 average = 'macro')*100))
    print("Recall (logistic regression): {:.2f}% (training) - {:.2f}% (testing)\n".format(recall_score(Y_train, 
                                                                                                       predictions_log_reg_train, 
                                                                                                       average = 'macro')*100,
                                                                                 recall_score(Y_test, 
                                                                                              predictions_log_reg_test, 
                                                                                              average = 'macro')*100))
    
    prediction_plotting(X_test, Y_test, predictions_log_reg_test, 'Logistic Regression')
    
    
def softmax_regression(X_train, X_train_not_red, X_test, Y_train, Y_test):
    
    # Softmax regression algorithm
    softmax_reg = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', C = 10)
    
    t_0 = time.time()
    softmax_reg.fit(X_train, Y_train) # Training
    print("\nTraining time (softmax regression prediction): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediciton time

    # Computing predictions on training and testing sets
    predictions_softmax_reg_train = softmax_reg.predict(X_train)
    
    # Measuring prediction time
    t_0 =time.time()

    predictions_softmax_reg_full_train = softmax_reg.predict(X_train_not_red)
    predictions_softmax_reg_test = softmax_reg.predict(X_test)

    print("Prediction time (softmax regression prediction): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediciton time

    # Computing accuracies
    training_accuracy_softmax_reg = np.sum(np.equal(Y_train, predictions_softmax_reg_train))/Y_train.shape[0] * 100
    testing_accuracy_softmax_reg = np.sum(np.equal(Y_test, predictions_softmax_reg_test))/Y_test.shape[0] * 100

    # Performance Metrics
    print("Confusion matrix (softmax regression):\n{}".format(confusion_matrix(Y_test, predictions_softmax_reg_test)))
    print("Accuracy (softmax regression): {:.2f}% (training) - {:.2f}% (testing)".format(training_accuracy_softmax_reg,
                                                                                 testing_accuracy_softmax_reg))
    print("Precision (softmax regression): {:.2f}% (training) - {:.2f}% (testing)".format(precision_score(Y_train, 
                                                                                                          predictions_softmax_reg_train, 
                                                                                                          average = 'macro')*100,
                                                                                 precision_score(Y_test, 
                                                                                                 predictions_softmax_reg_test, 
                                                                                                 average = 'macro')*100))
    print("Recall (softmax regression): {:.2f}% (training) - {:.2f}% (testing)".format(recall_score(Y_train, 
                                                                                                    predictions_softmax_reg_train, 
                                                                                                    average = 'macro')*100,
                                                                                 recall_score(Y_test, 
                                                                                              predictions_softmax_reg_test, 
                                                                                              average = 'macro')*100))

    prediction_plotting(X_test, Y_test, predictions_softmax_reg_test, 'Softmax Regression')
    
def svm(X_train, X_train_not_red, X_test, Y_train, Y_test):
    
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC

    # Linear SVM classification
    svm_clf = LinearSVC(C=10, loss = 'hinge', max_iter = 200000)
    
    t_0 = time.time()
    svm_clf.fit(X_train, Y_train)
    print("\nTraining time (SVM): {t_f:.3f} s\n".format(t_f = time.time() - t_0))
    
    # Computing predictions on training set
    y_pred_train = svm_clf.predict(X_train)

    t_0 = time.time()
    # Predicting on the full set of eigenvalues
    y_pred_full_train = svm_clf.predict(X_train_not_red)
    y_pred_test = svm_clf.predict(X_test)

    print("\nPrediction time (SVM): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediciton time

    # Computing accuracies
    training_accuracy_svm = metrics.accuracy_score(Y_train, y_pred_train)*100
    testing_accuracy_svm = metrics.accuracy_score(Y_test, y_pred_test)*100

    # Performance Metrics
    print("Confusion matrix (SVM):\n{}".format(confusion_matrix(Y_test, y_pred_test)))
    print("Accuracy (SVM): {:.2f}% (training) - {:.2f}% (testing)".format(training_accuracy_svm, testing_accuracy_svm))
    print("Precision (SVM): {:.2f}% (training) - {:.2f}% (testing)".format(precision_score(Y_train, y_pred_train, average = 'macro')*100,
                                                                                 precision_score(Y_test, y_pred_test, average = 'macro')*100))
    print("Recall (SVM): {:.2f}% (training) - {:.2f}% (testing)".format(recall_score(Y_train, y_pred_train, average = 'macro')*100,
                                                                                 recall_score(Y_test, y_pred_test, average = 'macro')*100))

    prediction_plotting(X_test, Y_test, y_pred_test, 'Support Vector Machine (SVM)')
    
def kNN(X_train, X_train_not_red, X_test, Y_train, Y_test):
    clf = KNeighborsClassifier(n_neighbors = 2)
    
    # Measuring training time
    t_0 = time.time()
    clf.fit(X_train, Y_train)

    # Computing predictions on training set
    y_pred_train = clf.predict(X_train)
    print("\nTraining time (k-Nearest Neighbors): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediciton time

    t_0 = time.time()

    # Predicting on the full set of eigenvalues
    y_pred_full_train = clf.predict(X_train_not_red)
    y_pred_test = clf.predict(X_test)

    print("\nPrediction time (k-Nearest Neighbors): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediciton time

    # Computing accuracies
    training_accuracy_kNN = metrics.accuracy_score(Y_train, y_pred_train)*100
    testing_accuracy_kNN = metrics.accuracy_score(Y_test, y_pred_test)*100

    # Performance Metrics
    print("Confusion matrix (k-Nearest Neighbors):\n{}".format(confusion_matrix(Y_test, y_pred_test)))
    print("Accuracy (k-Nearest Neighbors): {:.2f}% (training) - {:.2f}% (testing)".format(training_accuracy_kNN, testing_accuracy_kNN))
    print("Precision (k-Nearest Neighbors): {:.2f}% (training) - {:.2f}% (testing)".format(precision_score(Y_train, y_pred_train, average = 'macro')*100,
                                                                                 precision_score(Y_test, y_pred_test, average = 'macro')*100))
    print("Recall (k-Nearest Neighbors): {:.2f}% (training) - {:.2f}% (testing)".format(recall_score(Y_train, y_pred_train, average = 'macro')*100,
                                                                                 recall_score(Y_test, y_pred_test, average = 'macro')*100))


    prediction_plotting(X_test, Y_test, y_pred_test, 'k-Nearest Neighbors')

def decision_tree(X_train, X_train_not_red, X_test, Y_train, Y_test):
    
    # Selecting the depth hyperparameter as 7
    tree = DecisionTreeClassifier(max_depth = 7, random_state = 0)
    
    # Measuring training time
    t_0 = time.time()
    tree.fit(X_train, Y_train)
    print("\nTraining time (Decision Tree): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediction time
    
    # Computing predictions on training set
    y_pred_train = tree.predict(X_train)
    
    # Measuring prediction time
    t_0 = time.time()
    # Predicting on the full set of eigenvalues
    y_pred_full_train = tree.predict(X_train_not_red)
    y_pred_test = tree.predict(X_test)
    print("\nPrediction time (Decision Tree): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediction time

    # Computing accuracies
    training_accuracy_dt = metrics.accuracy_score(Y_train, y_pred_train)*100
    testing_accuracy_dt = metrics.accuracy_score(Y_test, y_pred_test)*100

    # Performance Metrics
    print("Confusion matrix (Decision Tree):\n{}".format(confusion_matrix(Y_test, y_pred_test)))
    print("Accuracy (Decision Tree): {:.2f}% (training) - {:.2f}% (testing)".format(training_accuracy_dt, testing_accuracy_dt))
    print("Precision (Decision Tree): {:.2f}% (training) - {:.2f}% (testing)".format(precision_score(Y_train, y_pred_train, average = 'macro')*100,
                                                                                 precision_score(Y_test, y_pred_test, average = 'macro')*100))
    print("Recall (Decision Tree): {:.2f}% (training) - {:.2f}% (testing)".format(recall_score(Y_train, y_pred_train, average = 'macro')*100,
                                                                                 recall_score(Y_test, y_pred_test, average = 'macro')*100))

    # Plotting prediction and ground truth
    prediction_plotting(X_test, Y_test, y_pred_test, 'Decision Tree')

def naive_bayes(X_train, X_train_not_red, X_test, Y_train, Y_test):
    from sklearn.naive_bayes import GaussianNB

    # Defining GNB objet
    gnb = GaussianNB()
    
    # Measuring training time
    t_0 = time.time()
    # Training GNB model
    gnb.fit(X_train, Y_train)
    print("Training time (Naïve Bayes prediction): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediciton time

    # Predicting on the training set
    y_pred_train = gnb.predict(X_train)

    # Measuring prediction time
    t_0 = time.time()

    # Predicting on the full set of eigenvalues
    y_pred_full_train = gnb.predict(X_train_not_red)
    y_pred_test = gnb.predict(X_test)

    print("Prediction time (Naïve Bayes prediction): {t_f:.3f} s\n".format(t_f = time.time() - t_0)) # Printing prediciton time

    # Computing accuracies
    training_accuracy_nb = metrics.accuracy_score(Y_train, y_pred_train)*100
    testing_accuracy_nb = metrics.accuracy_score(Y_test, y_pred_test)*100

    # Performance Metrics
    print("Confusion matrix (Naïve Bayes):\n{}".format(confusion_matrix(Y_test, y_pred_test)))
    print("Accuracy (Naïve Bayes): {:.2f}% (training) - {:.2f}% (testing)".format(training_accuracy_nb, testing_accuracy_nb))
    print("Precision (Naïve Bayes): {:.2f}% (training) - {:.2f}% (testing)".format(precision_score(Y_train, y_pred_train, average = 'macro')*100,
                                                                                 precision_score(Y_test, y_pred_test, average = 'macro')*100))
    print("Recall (Naïve Bayes): {:.2f}% (training) - {:.2f}% (testing)".format(recall_score(Y_train, y_pred_train, average = 'macro')*100,
                                                                                 recall_score(Y_test, y_pred_test, average = 'macro')*100))

    # Plotting prediction and ground truth
    prediction_plotting(X_test, Y_test, y_pred_test, 'Naïve Bayes')