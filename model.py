import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
random_state = 42

def display_model_results(model_results):
    '''
    This function takes in the model_results dataframe. This is a dataframe in tidy data format 
    containing the following information for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)
    The function returns a pivot table of those values for easy comparison of models, metrics, and samples. 
    '''
    # create a pivot table of the model_results dataframe
    # establish columns as the model_number, with index grouped by metric_type then sample_type, and values as score
    # the aggfunc uses a lambda to return each individual score without any aggregation applied
    return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type', 'sample_type'), 
                                     values='score',
                                     aggfunc=lambda x: x)

def get_best_model_results(model_results, metric_type='accuracy', n_models=3):
    '''
    This function takes in the model_results dataframe created in the Modeling stage of the 
    TelCo Churn analysis project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)

    The function identifies the {n_models} models with the highest scores for the given metric
    type, as measured on the validate sample.

    It returns a dataframe of information about those models' performance in the tidy data format
    (as described above). 

    The resulting dataframe can be fed into the display_model_results function for convenient display formatting.
    '''
    # create an array of model numbers for the best performing models
    # by filtering the model_results dataframe for only validate scores for the given metric type
    best_models = (model_results[(model_results.metric_type == metric_type) 
                               & (model_results.sample_type == 'validate')]
                                                 # sort by score value in descending order
                                                 .sort_values(by='score', 
                                                              ascending=False)
                                                 # take only the model number for the top n_models
                                                 .head(n_models)
                                                 .model_number
                                                 # and take only the values from the resulting dataframe as an array
                                                 .values)
    # create a dataframe of model_results for the models identified above
    # by filtering the model_results dataframe for only the model_numbers in the best_models array
    # TODO: make this so that it will return n_models, rather than only 3 models
    best_model_results = model_results[(model_results.model_number == best_models[0]) 
                                     | (model_results.model_number == best_models[1]) 
                                     | (model_results.model_number == best_models[2])]

    return best_model_results


def rfe_decision_tree(train,
                      validate, 
                      target, 
                      positive, 
                      model_number, 
                      model_info, 
                      model_results):

    # all available features
    all_features = [col for col in train.columns if 'enc_' in col or 'scaled_' in col]

    # separate each sample into x (features) and y (target) - for RFE
    x_train_rfe = train[all_features]
    y_train_rfe = train[target]

    for n_features in range(2,6):
        for max_depth in range(1,6):

            #####################################
            ### Recursive Feature Elimination ###
            #####################################

            # establish a decision tree classifier
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

            # create the rfe object
            rfe = RFE(clf, n_features_to_select=n_features)

            # fit the data using RFE
            rfe.fit(x_train_rfe, y_train_rfe)

            # get list of the column names for the selected features
            features = x_train_rfe.iloc[:,rfe.support_].columns.tolist()

            ##################
            ### Model Info ###
            ##################

            print(model_number)

            # create a new model number by adding 1 to the previous model number
            model_number += 1
            # establish the model type
            model_type = 'decision tree'

            # store info about the model

            # create a dictionary containing the features and hyperparamters used in this model instance
            dct = {'model_number': model_number,
                   'model_type': model_type,
                   'features': features,
                   'max_depth': max_depth}
            # append that dictionary to the model_info dataframe
            model_info = model_info.append(dct, ignore_index=True)

            ################
            ### Modeling ###
            ################

            # separate each sample into x (features) and y (target)
            x_train = train[features]
            y_train = train[target]

            x_validate = validate[features]
            y_validate = validate[target]


            # create the classifer

            # establish a decision tree classifier with the given max depth
            # set a random state for repoduceability
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

            # fit the classifier to the training data
            clf = clf.fit(x_train, y_train)

            #####################
            ### Model Results ###
            #####################

            ####### train #######

            # create prediction results for the model's performance on the train sample
            y_pred = clf.predict(x_train)
            sample_type = 'train'

            # get metrics

            # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'accuracy',
                   'score': sk.metrics.accuracy_score(y_train, y_pred)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'precision',
                   'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'recall',
                   'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'f1_score',
                   'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)


            ####### validate #######

            # create prediction results for the model's performance on the validate sample
            y_pred = clf.predict(x_validate)
            sample_type = 'validate'

            # get metrics

            # create dictionaries for each metric type for the validate sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'f1_score',
                   'score': sk.metrics.f1_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'accuracy',
                   'score': sk.metrics.accuracy_score(y_validate, y_pred)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'precision',
                   'score': sk.metrics.precision_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'recall',
                   'score': sk.metrics.recall_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True) 
            
    return model_number, model_info, model_results


def rfe_random_forest(train,
                      validate, 
                      target, 
                      positive, 
                      model_number, 
                      model_info, 
                      model_results): 
    
    # all available features
    all_features = [col for col in train.columns if 'enc_' in col or 'scaled_' in col]

    # separate each sample into x (features) and y (target) - for RFE
    x_train_rfe = train[all_features]
    y_train_rfe = train[target]

    for n_features in range(2, 6):
        for max_depth in range (2, 6):
            for min_samples_leaf in range(2, 3):
                
                #####################################
                ### Recursive Feature Elimination ###
                #####################################

                # establish a random forest classifier
                clf = RandomForestClassifier(max_depth=max_depth, 
                                             min_samples_leaf=min_samples_leaf,
                                             random_state=random_state)

                # create the rfe object
                rfe = RFE(clf, n_features_to_select=n_features)

                # fit the data using RFE
                rfe.fit(x_train_rfe, y_train_rfe)

                # get list of the column names for the selected features
                features = x_train_rfe.iloc[:,rfe.support_].columns.tolist()
                
                ##################
                ### Model Info ###
                ##################

                print(model_number)

                # create a new model number by adding 1 to the previous model number
                model_number += 1
                # establish the model type
                model_type = 'random forest'

                # store info about the model

                # create a dictionary containing the features and hyperparamters used in this model instance
                dct = {'model_number': model_number,
                       'model_type': model_type,
                       'features': features,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf}
                # append that dictionary to the model_info dataframe
                model_info = model_info.append(dct, ignore_index=True)
                
                ################
                ### Modeling ###
                ################

                # separate each sample into x (features) and y (target)
                x_train = train[features]
                y_train = train[target]

                x_validate = validate[features]
                y_validate = validate[target]


                # create the classifer

                # establish a random forest classifier 
                clf = RandomForestClassifier(max_depth=max_depth, 
                                             min_samples_leaf=min_samples_leaf,
                                             random_state=random_state)

                # fit the classifier to the training data
                clf = clf.fit(x_train, y_train)
                
                #####################
                ### Model Results ###
                #####################

                ####### train #######

                # create prediction results for the model's performance on the train sample
                y_pred = clf.predict(x_train)
                sample_type = 'train'

                # get metrics

                # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'accuracy',
                       'score': sk.metrics.accuracy_score(y_train, y_pred)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'precision',
                       'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'recall',
                       'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'f1_score',
                       'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)


                ####### validate #######

                # create prediction results for the model's performance on the validate sample
                y_pred = clf.predict(x_validate)
                sample_type = 'validate'

                # get metrics

                # create dictionaries for each metric type for the validate sample and append those dictionaries to the model_results dataframe
                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'f1_score',
                       'score': sk.metrics.f1_score(y_validate, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'accuracy',
                       'score': sk.metrics.accuracy_score(y_validate, y_pred)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'precision',
                       'score': sk.metrics.precision_score(y_validate, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'recall',
                       'score': sk.metrics.recall_score(y_validate, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True) 

    return model_number, model_info, model_results

def rfe_log_regression(train,
                      validate, 
                      target, 
                      positive, 
                      model_number, 
                      model_info, 
                      model_results):

    # all available features
    all_features = [col for col in train.columns if 'enc_' in col or 'scaled_' in col]

    # separate each sample into x (features) and y (target) - for RFE
    x_train_rfe = train[all_features]
    y_train_rfe = train[target]

    for n_features in range(2, 6):
        for c_value in [.001, .01, .1, 1, 10, 100, 1000]:
            
                #####################################
                ### Recursive Feature Elimination ###
                #####################################

                # establish a logistic regression classifier
                clf = LogisticRegression(C=c_value)

                # create the rfe object
                rfe = RFE(clf, n_features_to_select=n_features)

                # fit the data using RFE
                rfe.fit(x_train_rfe, y_train_rfe)

                # get list of the column names for the selected features
                features = x_train_rfe.iloc[:,rfe.support_].columns.tolist()
                
                ##################
                ### Model Info ###
                ##################

                print(model_number)

                # create a new model number by adding 1 to the previous model number
                model_number += 1
                # establish the model type
                model_type = 'logistic regression'

                # store info about the model

                # create a dictionary containing the features and hyperparamters used in this model instance
                dct = {'model_number': model_number,
                       'model_type': model_type,
                       'features': features,
                       'c_value': c_value}
                # append that dictionary to the model_info dataframe
                model_info = model_info.append(dct, ignore_index=True)
                
                ################
                ### Modeling ###
                ################

                # separate each sample into x (features) and y (target)
                x_train = train[features]
                y_train = train[target]

                x_validate = validate[features]
                y_validate = validate[target]
                
                # fit the classifier to the training data
                clf = clf.fit(x_train, y_train)
                
                #####################
                ### Model Results ###
                #####################

                ####### train #######

                # create prediction results for the model's performance on the train sample
                y_pred = clf.predict(x_train)
                sample_type = 'train'

                # get metrics

                # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'accuracy',
                       'score': sk.metrics.accuracy_score(y_train, y_pred)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'precision',
                       'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'recall',
                       'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'f1_score',
                       'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)


                ####### validate #######

                # create prediction results for the model's performance on the validate sample
                y_pred = clf.predict(x_validate)
                sample_type = 'validate'

                # get metrics

                # create dictionaries for each metric type for the validate sample and append those dictionaries to the model_results dataframe
                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'f1_score',
                       'score': sk.metrics.f1_score(y_validate, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'accuracy',
                       'score': sk.metrics.accuracy_score(y_validate, y_pred)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'precision',
                       'score': sk.metrics.precision_score(y_validate, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                       'sample_type': sample_type, 
                       'metric_type': 'recall',
                       'score': sk.metrics.recall_score(y_validate, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True) 
                
    return model_number, model_info, model_results