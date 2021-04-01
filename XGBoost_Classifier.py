# File Name: XGBoost_Classifier.py

# Load Libraries
import xgboost as xgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (14, 7)
marks = '-' * 100


def validation(model, scaled_X, y_trn, random_state):
    from sklearn.model_selection import GridSearchCV, RepeatedKFold
    # Passed arguments
    params_dict = {
        "objective": ['binary:logistic']
        , 'learning_rate': [0.1, 0.2, 0.3]
        , 'colsample_bytree': np.linspace(0.5, 0.9, 5)
        , "subsample": [0.4]
        , 'max_depth': [10]
        , 'n_estimators': [100]
        , "gamma": [0.5]
        , 'reg_alpha': [0.5]
        , 'reg_lambda': [0.5]
    }

    # Evaluation procedure
    c_v = RepeatedKFold(n_splits=10
                        , n_repeats=3
                        , random_state=random_state)

    # Grid search procedure
    grid_search = GridSearchCV(estimator=model
                               , param_grid=params_dict
                               , n_jobs=-1
                               # -1 means using all processors
                               , cv=c_v
                               , scoring='accuracy'
                               , error_score='raise'
                               )
    # Fit the best estimator on the training set
    grid_search.fit(scaled_X, y_trn)

    best_xgb_model = grid_search.best_estimator_
    # Best estimator - Estimator that was chosen by the search
    print("-- GridSearchCV best estimator result:\n\t %s" % best_xgb_model)

    return best_xgb_model


def xgb_sklearn(X_trn, X_tst, y_trn, y_tst, random_state):
    """ XGB method """
    """
        XGBoost is used for supervised learning problems, where we use the training data 
        (with multiple features) to predict a target variable. 
        Before we learn about trees specifically, let us start by reviewing the basic 
        elements in supervised learning.
        """
    scaled_X_train, scaled_X_test = scaling_data(X_trn, X_tst)
    #labeled_y_train = label_encoders(y_trn, 'transform')
    #labeled_y_test = label_encoders(y_tst, 'transform')
    # Instantiate the XGBoost Regressor --------------------------------------------
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Validate and get the best estimator
    best_xgb_model = validation(xgb_model, scaled_X_train, y_trn, random_state)

    # predict on the training set -------------------------------------------------
    predictions = best_xgb_model.predict(scaled_X_test)

    # Report the model performance
    # Performance report for testing data ---------------------
    test_data_list = performance_report(y_tst, predictions)
    # Performance report for training data ---------------------
    train_data_list = performance_report(y_trn, best_xgb_model.predict(scaled_X_train))

    metrics = ['Accuracy Score', 'Confusion Matrix Accuracy', 'F1 SCORE for test set']
    performance_df = pd.DataFrame({'Training performance': train_data_list
                                      , 'Testing performance': test_data_list}
                                  , index=metrics)
    print('\n-- Performance report:\n', performance_df)

    return best_xgb_model, predictions


def confusion_matrix_accuracy(cm):
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    return diagonal_sum / sum_of_all_elements


def performance_report(set1, set2):
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    acc = '{:.2f}'.format(accuracy_score(set1, set2) * 100.0)
    cm = '{:.2f}'.format(confusion_matrix_accuracy(confusion_matrix(set1, set2)) * 100.0)
    f1 = '{:.2f}'.format(f1_score(set1, set2, average='weighted') * 100.0)

    performance_list = [acc, cm, f1]

    return performance_list


def visualize_boosting(model):
    xgb.plot_tree(model, num_trees=0, rankdir='LR')
    plt.title('Visualize Boosting Trees')
    plt.show()


def plot_feature_importance(fi, nam):
    x_labels = list(fi.index)
    y_labels = list(fi['importance'])
    x = np.arange(1, len(x_labels) + 1, 1)

    plt.style.use('ggplot')
    # Format the plot
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.5

    ax.bar(x, fi['importance'], width)
    plt.axis([0, max(x) + width, 0, max(y_labels) + 0.1])
    ax.axhline(y=0, color='g')
    ax.axvline(x=0, color='g')
    ax.set_xlabel('Features', fontdict=font)
    ax.set_ylabel('Importance', fontdict=font)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    color = ['c', 'g', 'b', 'm']
    for i in x:
        ax.hlines(y=y_labels[i - 1]
                  , xmin=min(x) - 1
                  , xmax=i
                  , colors='b'
                  , linestyles='dashed')
        ax.text(x=i
                , y=y_labels[i - 1]
                , s='{:.4f}'.format(y_labels[i - 1])
                , ha='center'
                , va='bottom'
                , color='darkred'
                , fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Feature importance for ' + nam, fontdict=font)
    plt.tight_layout(True)
    plt.show()


def visualize_feature_importance(model):
    xgb.plot_importance(model)
    plt.title('Feature Importance')
    plt.show()


def plot_prediction(predicted, Z):
    plt.style.use('ggplot')
    predicted.plot(figsize=[14, 8],
                   x="prediction",
                   y="observed",
                   kind="scatter",
                   color='darkred')
    plt.title("Extreme Gradient Boosting: Prediction Vs Test Data",
              fontsize=18,
              color="darkgreen")
    plt.xlabel("Predicted Output", fontsize=18)
    plt.ylabel("Observed Output", fontsize=18)
    plt.plot(Z[:, 0], Z[:, 1], color="blue", lw=3)
    plt.show()


def label_encoders(labels, flag):
    from sklearn.preprocessing import LabelEncoder
    """ Encode target labels with value between 0 and n_classes-1. """
    label_encoder = LabelEncoder()
    # print(label_encoder.classes_) # to get the encoded labels
    if flag == 'transform':
        # Fit label encoder and return encoded labels.
        encoded_labels = label_encoder.fit_transform(labels)
    else:
        # Transform labels back to original encoding.
        encoded_labels = label_encoder.inverse_transform(labels)

    return encoded_labels


def scaling_data(x_train, x_test):
    """ Scaling or standardizing our training and test data """
    from sklearn.preprocessing import StandardScaler
    """
        -- Data standardization is the process of rescaling the attributes so that they have 
            mean as 0 and variance as 1.
        -- The ultimate goal to perform standardization is to bring down all the features to 
            a common scale without distorting the differences in the range of the values.
        -- In sklearn.preprocessing.StandardScaler(), centering and scaling happens independently 
            on each feature.
    """
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    """
        The "fit method" is calculating the mean and variance of each of the features present in our data. 
        The "transform" method is transforming all the features using the respective mean and variance.
    """
    scaled_x_train = scaler.fit_transform(x_train)
    """
        Using the "transform" method we can use the same mean and variance as it is calculated from our 
        training data to transform our test data. Thus, the parameters learned by our model using the 
        training data will help us to transform our test data.
    """
    scaled_x_test = scaler.transform(x_test)
    return scaled_x_train, scaled_x_test


def data_sets():
    from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_iris
    name, feature_labels, x, y = None, None, None, None
    print("\n Choose a data set? ------------------------------------")
    print(f" 1 : Brest Cancer dataset"
          f"\n 2 : Iris Dataset"
          f"\n ... Press any other key to EXIT.")
    choice = int(input("\n Enter your choice: "))

    if choice == 1:
        bc = load_breast_cancer()
        x, y = bc.data, bc.target
        feature_labels = bc.feature_names
        name = 'Brest Cancer dataset'
    elif choice == 2:
        iris = load_iris()
        x, y = iris.data, iris.target
        feature_labels = iris.feature_names
        name = 'Iris Dataset'

    else:
        print('\n.........................................')
        print('No Data has been chosen! ..... Exit .....')
        print('.........................................\n')
        exit()

    return name, feature_labels, x, y


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    import time

    while True:
        print(marks)
        name, labels, features, target = data_sets()

        random_state = 123

        X_train, X_test, y_train, y_test = train_test_split(features
                                                            , target
                                                            , test_size=0.2
                                                            , random_state=random_state)
        start = time.time()

        """
            XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses 
            a gradient boosting framework. Can be used to solve regression, classification, 
            ranking, and user-defined prediction problems.

            XGBoost is an implementation of gradient boosted decision trees designed for speed 
            and performance.
            """
        best_model, y_predicted = xgb_sklearn(X_train, X_test, y_train, y_test, random_state)

        end = time.time()
        print(f'\nExecution Time: {(end - start):.1f} seconds --> {(end - start) / 60: .1f} minutes')

        # Visualize Boosting Trees
        visualize_boosting(best_model)

        # Visualize Feature Importance
        feature_imp = pd.DataFrame({'importance': best_model.feature_importances_}
                                   , index=list(labels))
        # print('\n-- Feature Importance:\n', feature_imp.sort_values(by=['importance'], ascending=False))
        # visualize_feature_importance(best_model)
        plot_feature_importance(feature_imp.nlargest(10, 'importance').sort_values(by=['importance']
                                                                                   , ascending=False), name)

        # Plot prediction vs y true
        import statsmodels.api as sm

        predictions_vs_observed = pd.DataFrame({"prediction": y_predicted, "observed": y_test})
        print('-- Predictions vs Observed \n', predictions_vs_observed.head())
        # ndarray.flatten(order='C'): Return a copy of the array collapsed into one dimension.
        # A lowess function that outs smoothed estimates of endog at the given
        # exog values from points (exog, endog)
        z = sm.nonparametric.lowess(np.array(y_predicted).flatten(), np.array(y_test).flatten())
        plot_prediction(predictions_vs_observed, z)
