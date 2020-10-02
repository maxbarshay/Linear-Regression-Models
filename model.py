# Packages
import pandas as pd
import numpy as np
import math
import itertools
import random

# Options
pd.set_option("display.max_rows", 100)

# Read in data (aggregated)
iowa = pd.read_csv('C:/Users/Mason/Desktop/Git Repositories/DATA401/Project1/agg_data.csv')

# Clean data
iowa = iowa.dropna()  # drop all NA values
iowa["Date"] = pd.to_datetime(iowa["Date"])
iowa["Month"] = iowa["Date"].dt.month
iowa["Year"] = iowa["Date"].dt.year

# Choosing variables that make sense:
all_features = ['Zip Code', 'Population', 'State Bottle Cost', 'State Bottle Retail', 'Month', 'Year']
response_var = 'Volume Sold (Liters)'
y = iowa[response_var].to_numpy()

numeric_vars = ['Population', 'State Bottle Cost', 'State Bottle Retail']
categorical_vars = ['Year', 'Zip Code', 'Month']

# Convert every categorical variable into 'category' type (this helps later when dummifying variables)
for variable in categorical_vars:
    iowa[variable] = iowa[variable].astype('category')


def get_X_matrix(feature_list, iowa, categorical_vars):
    """Arguments:
            feature_list:     a list of features being used in the model
            iowa:             the original data
            categorical_vars: a list of which variables in the original data are categorical

       Returns: a 2D numpy array, with categorical variables turned into indicators"""

    this_data = iowa[feature_list]  # keep just our features

    for feature in feature_list:  # for every feature in our list of features
        if feature in categorical_vars:  # if that feature is categorical:
            dummified = pd.get_dummies(this_data[feature], drop_first=True)  # dummify the categorical variable
            this_data = pd.concat([this_data, dummified], axis=1)  # put the new columns into the data
            cols_in_data = list(this_data.columns)  # get a list of variables in our new dummified data
            cols_in_data.remove(feature)  # remove original variable, keeping dummy variables
            this_data = this_data[cols_in_data]  # remove the original feature

    nrows = len(this_data)
    intercept_col = np.array([1] * nrows).reshape(nrows, 1)
    data_array = this_data.to_numpy()
    return np.concatenate((intercept_col, data_array), 1)


def get_num_parameters(feature_list, categorical_vars, iowa):
    """Arguments:
            feature_list:     a list of column names being used in the model
            categorical_vars: a list of which variables in the original data are categorical
            iowa:             the original data
       Returns: an integer p, equal to the number of parameters being estimated in the model"""
    p = 1  # initialize p, accounting for intercept
    for feature in feature_list:  # for every feature in this model
        if feature in categorical_vars:  # if the feature is categorical
            p += len(iowa[feature].unique()) - 1  # (number of categories - 1)
        else:  # if the feature is numeric
            p += 1  # then there's just one coefficient to estimate
    return p


def get_Beta_vector(X, y):
    """Arguments:
            X: the design matrix as a numpy array
            y: the response variable as a 1 x n numpy array
       Returns: a numpy array of length p of coefficients for all predictor variables"""
    X_t = np.transpose(X)
    inside = X_t.dot(X)
    # Problem happens here. It say's it's a singular matrix, which means that it's a square matrix
    # that's not invertible. Adding column for intercept didn't help. Possible there's multicollinearity.
    # It's only happening when 'County' is a predictor, by itself or with others
    left = np.linalg.inv(X_t.dot(X))
    right = X_t.dot(y)
    result = left.dot(X_t.dot(y))
    return result


def get_MSE(y, y_hat):
    """Arguments:
            y:     the response variable as a 1 x n numpy array
            y_hat: the predicted values of the response variable as a 1 x n numpy array
       Returns: the Sum of Squared Error, a float"""
    return (1 / len(y)) * sum([(y[i] - y_hat[i]) ** 2 for i in range(len(y))])


def get_AIC(y, y_hat, p):
    """Arguments:
            y:     the response variable as a 1 x n numpy array
            y_hat: the predicted values of the response variable as a 1 x n numpy array
            p:     the number of parameters being estimated in the model, an integer
       Returns: the AIC, a float"""
    return get_MSE(y, y_hat) + (2 * p)


def get_BIC(y, y_hat, p):
    """Arguments:
            y:     the response variable as a 1 x n numpy array
            y_hat: the predicted values of the response variable as a 1 x n numpy array
            p:     the number of parameters being estimated in the model, an integer
       Returns: the BIC, a float"""
    return get_MSE(y, y_hat) + (p * math.log(len(y)))


def print_models(models):
    """This function just prints out the models in a way that looks nice.
       I wrote this so I didn't have to make a 'Model' object with a __repr__"""
    # model is defined as [[list of features], p, AIC, BIC, MSE]
    print("-" * 20)
    if models[0][2] is not None:  # if these models have been scored
        # Sort models by average of AIC, BIC, MSE just to make it easier to comb through them later
        sorted_models = sorted(models, key = lambda model: avg_of_list([model[2], model[3], model[4]]))
        for model in sorted_models:
            print("Features:", ", ".join(model[0]))
            print("p =", model[1])
            print("cross-val AIC =", model[2])
            print("cross-val BIC =", model[3])
            print("cross-val MSE =", model[4])
            print("-" * 20)

    else:
        for model in models:
            print("Features:", ", ".join(model[0]))
            print("p =", model[1])
            print("-" * 20)


def get_forward_stepwise_models(all_features, categorical_vars, iowa):
    """Arguments:
            all_features:     a list of all feature names
            categorical_vars: a list of which variables in the original data are categorical
            iowa:             the original data frame
       Returns: a list of models, where each model has the following format:
                [[list of features], p, AIC, BIC]"""
    models = []
    for i in range(1, len(all_features) + 1):
        this_model = [None] * 5
        vars_in_model = all_features[:i]  # only take first i features
        p = get_num_parameters(vars_in_model, categorical_vars, iowa)

        this_model[0] = vars_in_model  # save variables being used
        this_model[1] = p  # save number of parameters being estimated
        this_model[2] = []  # initialize empty lists for AIC, BIC
        this_model[3] = []
        this_model[4] = []

        models.append(this_model)
    return models


def get_backward_stepwise_models(all_features, categorical_vars, iowa):
    """Arguments:
            all_features:     a list of all feature names
            categorical_vars: a list of which variables in the original data are categorical
            iowa:             the original data frame
       Returns: a list of models, where each model has the following format:
                [[list of features], p, AIC, BIC]"""
    models = []
    for i in range(len(all_features), 0, -1):
        this_model = [None] * 5
        vars_in_model = all_features[:i]  # only take first i features
        p = get_num_parameters(vars_in_model, categorical_vars, iowa)

        this_model[0] = vars_in_model  # save variables being used
        this_model[1] = p  # save number of parameters being estimated
        this_model[2] = []  # initialize empty lists for AIC, BIC
        this_model[3] = []
        this_model[4] = []

        models.append(this_model)
    return models


def get_best_subsets_models(all_features, categorical_vars, iowa):
    """Arguments:
            all_features:     a list of all feature names
            categorical_vars: a list of which variables in the original data are categorical
            iowa:             the original data frame
       Returns: a list of models, where each model has the following format:
                [[list of features], p, AIC, BIC, MSE]"""
    models = []
    # generating subsets:
    for i in range(1, len(all_features) + 1):
        subsets = itertools.combinations(all_features, i)
        for subset in subsets:
            this_model = [None] * 5
            vars_in_model = list(subset)
            p = get_num_parameters(vars_in_model, categorical_vars, iowa)

            this_model[0] = vars_in_model  # save variables being used
            this_model[1] = p  # save number of parameters being estimated
            this_model[2] = []  # initialize empty lists for AIC, BIC
            this_model[3] = []
            this_model[4] = []

            models.append(this_model)
    return models


def get_fold_indices(k, iowa, categorical_vars):

    # One possible function to use: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    # Another option: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split

    # I may need to change this function. It works as-is, but it's creating k 'folds' really, it is creating
    # essentially random training and test sets. So, the cross-validated metrics are pretty different between
    # different runs of the program

    """Arguments:
            k:     number of partitions (folds) to create in the data
            iowa: number of rows in the data you want to partition
            categorical_vars: a list of which variables in the original data are categorical
       Returns: a list of lists, where each list has the indices of a fold"""
    nrow_test = len(iowa) // k
    nrow_train = len(iowa) - nrow_test

    partitions = []
    # output looks like [[[train_indices], [test_indices]], [[train_indices], [test_indices]]]
    done = False
    for i in range(k):  # creating k different splits of the data into test and train
        these_train_indices = []
        these_test_indices = []
        for cat_var in categorical_vars:  # for every categorical variable in the data
            for level in iowa[cat_var].unique():  # for every level of that categorical variable
                this_level_indices = list(iowa[iowa[cat_var] == level].index)  # keep only data with that level
                this_selection = random.sample(this_level_indices, 1)  # pick a single index of this level
                these_train_indices.append(this_selection[0])

        # At this point all levels of all categorical variables are accounted for, so I can finish picking the points
        # Now these_train_indices has 241 values, so there are 241 levels of categorical variables in total
        num_rows_left_to_pick = nrow_train - len(these_train_indices)

        # listing rows I haven't picked yet:
        indices_left = [num for num in range(0, len(iowa)) if num not in these_train_indices]

        # now that every level of every factor is accounted for, I'll pick the rest of the partition randomly
        rest_of_partition = random.sample(indices_left, num_rows_left_to_pick)
        these_train_indices += rest_of_partition  # add these indices to partition indices

        # Test set is just rows that aren't in the training set
        these_test_indices = [num for num in range(0, len(iowa)) if num not in these_train_indices]

        partitions.append([these_train_indices, these_test_indices])  # add this paritition to list of partitions

    test = 3
    return partitions


def avg_of_list(in_list):
    """Arguments:
            in_list: a list of floats or integers
       Returns: the average of those floats/integers"""
    return sum(in_list) / len(in_list)


def calc_cross_val_scores(iowa, categorical_vars, k, models):
    # select fraction of observations per county so that every county is in training data
    # so i need to rewrite my get_fold_indices function
    """Arguments:
            iowa:             the original data frame
            categorical_vars: a list of which variables in the original data are categorical
            k:                number of folds to use ('k-fold' cross validation)
            models:           list of models to calculate scores for, with the following format:
                              [[list of features], [B_hat vector], p, [], [], []]
       Returns: the same list of models passed in, but with cross-validated AIC, BIC, and MSE
                replacing the three 'None's in the model:
                [[list of features], p, AIC, BIC, MSE]"""

    data_folds = get_fold_indices(k, iowa, categorical_vars)

    for fold in data_folds:  # for every fold that we have (k folds)
        # fold is of the form [train_indices, test_indices]
        train_indices = fold[0]
        test_indices = fold[1]
        iowa_test = iowa.iloc[test_indices]  # partition test data set
        iowa_train = iowa.iloc[train_indices]  # partition data train set

        for model in models:  # for every model inputted to the function
            X_train = get_X_matrix(model[0], iowa_train, categorical_vars)  # get X training matrix

            # If statement below is a debugging thing, ignore
            if np.linalg.det(np.transpose(X_train).dot(X_train)) == 0:  # if determinant is 0
                print(model[0])  # then that means the matrix is non-invertible, which is a problem

            y_train = iowa_train['Volume Sold (Liters)'].to_numpy()
            Beta_vector = get_Beta_vector(X_train, y_train)  # calculate coefficients

            X_test = get_X_matrix(model[0], iowa_test, categorical_vars)  # get X test matrix
            y_test = iowa_test['Volume Sold (Liters)'].to_numpy()  # true values of response
            y_hat_test = X_test.dot(Beta_vector)  # calculate predicted values for this model

            model[2].append(get_AIC(y_test, y_hat_test, model[1]))  # add this AIC to the list of AICs (to be averaged later)
            model[3].append(get_BIC(y_test, y_hat_test, model[1]))  # add this BIC to the list of BICs
            model[4].append(get_MSE(y_test, y_hat_test))  # add this MSE to the list of MSEs

    # Average AICs, BICs, MSEs for every model
    models = [[model[0], model[1], avg_of_list(model[2]), avg_of_list(model[3]), avg_of_list(model[4])] for model in models]

    return models


raw_models = get_best_subsets_models(all_features, categorical_vars, iowa)
scored_models = calc_cross_val_scores(iowa, categorical_vars, 4, raw_models)
print_models(scored_models)



