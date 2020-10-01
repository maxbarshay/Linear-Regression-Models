# Packages
import pandas as pd
import numpy as np
import math
import itertools

# Options
pd.set_option("display.max_rows", 100)

# Read in data (aggregated)
iowa = pd.read_csv('C:/Users/Mason/Desktop/Git Repositories/DATA401/Project1/agg_data.csv')

# Clean data
iowa = iowa.dropna()  # drop all NA values

# Choosing variables that make sense:
all_features = ['County', 'Year', 'Bottles Sold', 'Sale (Dollars)', 'Population']
response_var = 'Volume Sold (Liters)'
y = iowa[response_var].to_numpy()

numeric_vars = ['Bottles Sold', 'Sale (Dollars)', 'Population']
categorical_vars = ['County', 'Year']

# model is defined as [[list of features], [B_hat vector], p, AIC, BIC]


def get_X_matrix(feature_list, iowa, categorical_vars):
    """Arguments:
            feature_list:     a list of features being used in the model
            iowa:             the original data
            categorical_vars: a list of which variables in the original data are categorical

       Returns: a 2D numpy array, with categorical variables turned into indicators"""

    this_data = iowa[feature_list]  # keep just our features

    for feature in feature_list:  # for every feature in our list of features
        if feature in categorical_vars:  # if that feature is categorical:
            dummified = pd.get_dummies(this_data[feature])  # dummify the categorical variable
            this_data = pd.concat([this_data, dummified], axis=1)  # put the new columns into the data
            cols_in_data = list(this_data.columns)  # get a list of variables in our new dummified data
            cols_in_data.remove(feature)  # remove original variable, keeping dummy variables
            this_data = this_data[cols_in_data]  # remove the original feature

    return this_data.to_numpy()


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
    return np.linalg.inv(X_t.dot(X)).dot(X_t.dot(y))


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
    # model is defined as [[list of features], [B_hat vector], p, AIC, BIC]
    for model in models:
        print("Features:", ", ".join(model[0]))
        print("p =", model[2])
        print("AIC =", model[3])
        print("BIC =", model[4])
        print("-" * 20)


def get_forward_stepwise_models(all_features, categorical_vars, iowa):
    """Arguments:
            all_features:     a list of all feature names
            categorical_vars: a list of which variables in the original data are categorical
            iowa:             the original data frame
       Returns: a list of models, where each model has the following format:
                [[list of features], [B_hat vector], p, AIC, BIC]"""
    models = []
    for i in range(1, len(all_features) + 1):
        this_model = [None] * 5
        vars_in_model = all_features[:i]  # only take first i features
        X = get_X_matrix(vars_in_model, iowa, categorical_vars)  # get X matrix
        Beta_vector = get_Beta_vector(X, y)  # calculate coefficients
        p = get_num_parameters(vars_in_model, categorical_vars, iowa)
        pred_vals = X.dot(Beta_vector)  # calculate predicted values for this model
        AIC = get_AIC(y, pred_vals, p)
        BIC = get_BIC(y, pred_vals, p)

        this_model[0] = vars_in_model  # save variables being used
        this_model[1] = Beta_vector  # save coefficients
        this_model[2] = p
        this_model[3] = AIC  # save AIC
        this_model[4] = BIC  # save BIC

        models.append(this_model)
    return models


def get_backward_stepwise_models(all_features, categorical_vars, iowa):
    """Arguments:
            all_features:     a list of all feature names
            categorical_vars: a list of which variables in the original data are categorical
            iowa:             the original data frame
       Returns: a list of models, where each model has the following format:
                [[list of features], [B_hat vector], p, AIC, BIC]"""
    models = []
    for i in range(len(all_features), 0, -1):
        this_model = [None] * 5
        vars_in_model = all_features[:i]  # only take first i features
        X = get_X_matrix(vars_in_model, iowa, categorical_vars)  # get X matrix
        Beta_vector = get_Beta_vector(X, y)  # calculate coefficients
        p = get_num_parameters(vars_in_model, categorical_vars, iowa)
        pred_vals = X.dot(Beta_vector)  # calculate predicted values for this model
        AIC = get_AIC(y, pred_vals, p)
        BIC = get_BIC(y, pred_vals, p)

        this_model[0] = vars_in_model  # save variables being used
        this_model[1] = Beta_vector  # save coefficients
        this_model[2] = p
        this_model[3] = AIC  # save AIC
        this_model[4] = BIC  # save BIC

        models.append(this_model)
    return models


def get_best_subsets_models(all_features, categorical_vars, iowa):
    """Arguments:
            all_features:     a list of all feature names
            categorical_vars: a list of which variables in the original data are categorical
            iowa:             the original data frame
       Returns: a list of models, where each model has the following format:
                [[list of features], [B_hat vector], p, AIC, BIC]"""
    models = []
    # generating subsets:
    for i in range(1, len(all_features) + 1):
        subsets = itertools.combinations(all_features, i)
        for subset in subsets:
            this_model = [None] * 5
            vars_in_model = list(subset)
            X = get_X_matrix(vars_in_model, iowa, categorical_vars)  # get X matrix
            Beta_vector = get_Beta_vector(X, y)  # calculate coefficients
            p = get_num_parameters(vars_in_model, categorical_vars, iowa)
            pred_vals = X.dot(Beta_vector)  # calculate predicted values for this model
            AIC = get_AIC(y, pred_vals, p)
            BIC = get_BIC(y, pred_vals, p)

            this_model[0] = vars_in_model  # save variables being used
            this_model[1] = Beta_vector  # save coefficients
            this_model[2] = p
            this_model[3] = AIC  # save AIC
            this_model[4] = BIC  # save BIC

            models.append(this_model)
    return models


models = get_best_subsets_models(all_features, categorical_vars, iowa)
print_models(models)


