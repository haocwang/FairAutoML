import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import warnings
warnings.filterwarnings("ignore")
import autosklearn.classification
from utils import generate_formatted_dataframe, train_test_split, precheck_data_fairness, postcheck_data_fairness, compute_metrics, model_and_find_threshold
from fair_preprocessing import disparateImpactRemover, learningFairRepresentation, optimPreproc
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.datasets import StandardDataset
from aif360.metrics.classification_metric import ClassificationMetric
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from tqdm import tqdm
# from IPython.display import Markdown, display

class FairAutoML():
    def __init__(self, input_columns, label_name, favorable_classes, 
                 protected_attribute_names, privileged_classes,
                 privileged_groups, unprivileged_groups,
                 categorical_features=[], features_to_keep=[], 
                 features_to_drop=[], is_valid=True, na_values=[],
                custom_preprocessing=None, metadata=None):
        '''
        @usage:
            to initialize a fairautoml model
        @params:
            - input_columns: list, containing strings of the name of input columns
            - label_name: str, the name of output column
            - favorable_classes:(list or function): Label values which are
                considered favorable or a boolean function which returns `True`
                if favorable. All others are unfavorable. Label values are
                mapped to 1 (favorable) and 0 (unfavorable) if they are not
                already binary and numerical.
            - protected_attribute_names: list, containing strings of the name of the protected columns
            - privileged_classes: privileged_classes (list(list or function)): Each element is
                a list of values which are considered privileged or a boolean
                function which return `True` if privileged for the corresponding
                column in `protected_attribute_names`. All others are
                unprivileged. Values are mapped to 1 (privileged) and 0
                (unprivileged) if they are not already numerical.
                1) e.g. for age above 25 to be the previledged class, you can write:
                protected_attribute_names=['age']
                privileged_classes=[lambda x: x >= 25]
                2) e.g. for male whose age above 25 to be the previledged class, you can write:
                protected_attribute_names=['sex', 'age']
                privileged_classes=[['male'], lambda x: x >= 25]
            - privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attribute_names`. Each `dict`
                element describes a single group. See examples for more details.
                unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.
            - unprivileged_groups (list(dict)): similar as privileged_groups
            - categorical_features (optional): list, containing strings of the name of the categorical columns
            - features_to_keep (optional, list): Column names to keep. All others
                are dropped except those present in `protected_attribute_names`,
                `categorical_features`, `label_name` or `instance_weights_name`.
                Defaults to all columns if not provided.
            - features_to_drop: list (default: None), containing strings of the name of columns which to be dropped
            - is_valid: if the dataset is valid as in StandardDataset()
            - na_values (optional): Additional strings to recognize as NA.
            - metadata (optional): Additional metadata to append.
        '''
        self.input_columns = input_columns
        self.label_name = label_name
        self.favorable_classes = favorable_classes
        self.protected_attribute_names = protected_attribute_names
        self.privileged_classes = privileged_classes
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.categorical_features = categorical_features
        self.features_to_keep = features_to_keep
        self.features_to_drop = features_to_drop
        self.is_valid = is_valid
        self.na_values = na_values
        self.custom_preprocessing = custom_preprocessing
        self.metadata = metadata
        self.preproc_name = None
        self.classifier = None
        self.best_ultimate_thres = None
        self.best_acc = 0
        self.fairness_arr = None
        self.acc_arr = None
        self.no_postproc = None
        #self.cpp = None # for post-processing
    
    def fit(self, dataset, dataset_metric, dataset_metric_threshold,
            classifier_metric, optim_options = None,
            time_left_for_this_task=120, per_run_time_limit=30, train_split_size = 0.8,
            verbose=True):
        '''
        @usage:
            fit the automl model to training data
        @params:
            - dataset_orig_train: original training data after splitting and 
                                  transformation
            - dataset_metric: str. The fairness metric for checking data fairness before modeling
            - dataset_metric_threshold
            
            - classifier_metric: str. 
            - time_left_for_this_task
            - per_run_time_limit
            - verbose: bool, True to print results in the process
        '''
        # 1. Transform data into StandardDataset if not valid dataset
        if not self.is_valid:
            dataset_standardized = generate_formatted_dataframe(dataset, label_name = self.label_name, \
                                                                favorable_classes= self.favorable_classes, \
                                                                protected_attribute_names = self.protected_attribute_names, \
                                                                privileged_classes= self.privileged_classes,\
                                                                categorical_features= self.categorical_features, \
                                                                features_to_keep= self.features_to_keep, \
                                                                features_to_drop= self.features_to_drop,\
                                                                na_values = self.na_values, \
                                                                custom_preprocessing = self.custom_preprocessing, \
                                                                metadata = self.metadata)
        else:
            dataset_standardized = dataset
            
        dataset_orig_train, dataset_orig_valid = train_test_split(dataset_standardized, train_split_size)                         
        
        # 2. Before preprocessing and fitting model：check whether dataset fairness metric for orig train is under the threshold
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                                     unprivileged_groups=self.unprivileged_groups,
                                                     privileged_groups=self.privileged_groups)
        
        # display(Markdown("#### Original training dataset"))
        no_preproc = precheck_data_fairness(dataset_metric, dataset_metric_threshold, metric_orig_train)
        # save classfier_metric for plot function.
        self.classifier_metric = classifier_metric  
        
        # 3. Do Preprocessing if the threshold is not met.
        if no_preproc: # If no need for preprocessing, standard automl
            # build model and find best probability threshold
            self.classifier, self.best_ultimate_thres = \
            model_and_find_threshold(dataset_orig_train, dataset_orig_valid, \
                                     self.unprivileged_groups, self.privileged_groups, \
                                     no_preproc, classifier_metric, time_left_for_this_task, per_run_time_limit)
            # no need for postproc if no need for preproc
            no_postproc = True
         
        else: # if need preprocessing: first preprocessing, then standard ml on transformed data

            print("--------------Start Preprocessing--------------")
            # Use all 3 preprocessing methods with tuning the first 2 methods: 
                # optimPrepro(dataset, optim_options)
                # disparate_impact_remover(data, repair_level: num in [0,1])
                # learningFairRepresentation(dataset, unprivileged_groups, privileged_groups, hreshold = 0.5, **kwargs)   
                
            # No grid search over optimized preprocessing
            if optim_options:
                try:
                    # Perform optimized preprocessing
                    preproc_name = 'Optimized Preprocessing'
                    dataset_transf_train = optimPreproc(dataset_orig_train, optim_options)
                    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                             unprivileged_groups=self.unprivileged_groups,
                                             privileged_groups=self.privileged_groups)
                    postcheck_data_fairness(dataset_metric, metric_transf_train, preproc_name)

                    # model the transformed train data and find the optimal classifier with its acc
                    self.classifier, self.best_ultimate_thres, self.best_acc, self.fairness_arr, self.acc_arr, \
                    self.no_postproc, self.lo, self.hi =\
                    model_and_find_threshold(dataset_transf_train, dataset_orig_valid, \
                                             self.unprivileged_groups, self.privileged_groups, \
                                             no_preproc, classifier_metric, time_left_for_this_task, per_run_time_limit)
                    self.preproc_name = preproc_name
                except:
                    pass
                
                
            # Grid search over disparateImpactRemover over repair_level
            repair_level = np.linspace(0.1,1,10)
            for i in repair_level:
                try:
                    preproc_name = 'Disparate Impact remover with repair_level = {}'.format(i)
                    dataset_transf_train = disparateImpactRemover(dataset_orig_train, i)
                    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                             unprivileged_groups=self.unprivileged_groups,
                                             privileged_groups=self.privileged_groups)
                    postcheck_data_fairness(dataset_metric, metric_transf_train, preproc_name)

                    # model the transformed train data and find the optimal classifier with its acc
                    classifier, best_ultimate_thres, best_acc, fairness_arr, acc_arr, no_postproc, lo, hi = \
                    model_and_find_threshold(dataset_transf_train, dataset_orig_valid, \
                                             self.unprivileged_groups, self.privileged_groups, \
                                             no_preproc, classifier_metric, time_left_for_this_task, per_run_time_limit)
                    
                    if best_acc > self.best_acc:
                        self.preproc_name = preproc_name
                        self.classifier = classifier
                        self.best_ultimate_thres = best_ultimate_thres
                        self.best_acc = best_acc
                        self.fairness_arr = fairness_arr
                        self.acc_arr = acc_arr
                        self.no_postproc = no_postproc
                        self.lo = lo
                        self.hi = hi
                except:
                    pass
            
            # Grid search over lfr over transform threshold
            lfr_thres = np.linspace(0.1,0.9,5)
            for i in lfr_thres:
                try:
                    preproc_name = 'LFR with transform threshold = {}'.format(i)
                    dataset_transf_train = learningFairRepresentation(dataset_orig_train, self.unprivileged_groups, \
                                                                      self.privileged_groups, threshold = i)
                    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                             unprivileged_groups=self.unprivileged_groups,
                                             privileged_groups=self.privileged_groups)
                    postcheck_data_fairness(dataset_metric, metric_transf_train, preproc_name)

                    # model the transformed train data and find the optimal classifier with its acc
                    classifier, best_ultimate_thres, best_acc, fairness_arr, acc_arr, no_postproc, lo, hi = \
                    model_and_find_threshold(dataset_transf_train, dataset_orig_valid, \
                                             self.unprivileged_groups, self.privileged_groups, \
                                             no_preproc, classifier_metric, time_left_for_this_task, per_run_time_limit)

                    if best_acc > self.best_acc:
                        self.preproc_name = preproc_name
                        self.classifier = classifier
                        self.best_ultimate_thres = best_ultimate_thres
                        self.best_acc = best_acc
                        self.fairness_arr = fairness_arr
                        self.acc_arr = acc_arr                
                        self.no_postproc = no_postproc
                        self.lo = lo
                        self.hi = hi
                except:
                    pass
                    
            
        # Do post-processing if needed    
        if not self.no_postproc: # if post-processing is needed
            ################# need further development ############
            print("The preprocessing doesn't work well. Will Implement postprocessing in the future version")
#             print("--------------Start Postprocessing-------------")
#             # Learn parameters to equalize odds and apply to create a new dataset
#             # the user will be asked to specify cost_constraint, and randseed
#             cost_constraint, randseed = input("e.g. 'fnr', 'fpr', 'weighted' "), input("e.g. 12345679")
#             cpp = CalibratedEqOddsPostprocessing(privileged_groups = self.privileged_groups,
#                                                  unprivileged_groups = self.unprivileged_groups,
#                                                  cost_constraint=cost_constraint,
#                                                  seed=randseed)
#             self.cpp = cpp.fit(dataset_orig_valid, dataset_transf_val_pred)
            
    
    def plot(self):
        class_thresh_arr = np.linspace(0.01, 0.99, 50)
        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.plot(class_thresh_arr, self.acc_arr)
        ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)


        ax2 = ax1.twinx()
        ax2.plot(class_thresh_arr, self.fairness_arr, color='r')
        ax2.set_ylabel(self.classifier_metric, color='r', fontsize=16, fontweight='bold')
        criterion = np.logical_and(self.fairness_arr > self.lo, self.fairness_arr < self.hi)
        collection = collections.BrokenBarHCollection.span_where(np.linspace(0.01, 0.99,50), ymin= self.lo, ymax=self.hi, \
                                                                 where=criterion, facecolor='orange', alpha=0.5)
        ax2.add_collection(collection)
        ax2.axvline(self.best_ultimate_thres, color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)
        plt.title('Balance Accuracy V.S Fairness Metric.', loc ='left')
        plt.title('Orange area: Fair Classfication Threshold Range', loc='right')
        return fig

    
    def evaluate(self, dataset_orig_test, verbose=True):
        assert self.classifier, 'There is no model to use. Please fit the model first.'
        
        best_ultimate_thres = self.best_ultimate_thres

        # Transform into standardized dataframe
        if not self.is_valid:
            # dataset_orig_test[self.label_name] = self.favorable_classes[0]
            dataset_transf_test = generate_formatted_dataframe(dataset_orig_test, label_name=self.label_name, \
                                                            favorable_classes=self.favorable_classes, \
                                                            protected_attribute_names=self.protected_attribute_names, \
                                                            privileged_classes=self.privileged_classes,\
                                                            categorical_features=self.categorical_features, \
                                                            features_to_keep=self.features_to_keep, \
                                                            features_to_drop=self.features_to_drop,\
                                                            na_values=self.na_values, \
                                                            custom_preprocessing=self.custom_preprocessing, \
                                                            metadata=self.metadata)
            
            print("Data has been transformed into standardized dataframe.")
        else:
            dataset_transf_test = dataset_orig_test
            
        dataset_transf_test_pred = dataset_transf_test.copy(deepcopy=True)
        X_test = dataset_transf_test_pred.features
        y_test = dataset_transf_test_pred.labels
        
        # Predict_proba on test data
        pos_ind = dataset_transf_test_pred.favorable_label
        y_pred = self.classifier.predict_proba(X_test)[:,int(pos_ind)].reshape(-1,1)

        fav_inds = y_pred > best_ultimate_thres
        dataset_transf_test_pred.labels[fav_inds] = dataset_transf_test_pred.favorable_label
        dataset_transf_test_pred.labels[~fav_inds] = dataset_transf_test_pred.unfavorable_label
        
        metric_test_aft = compute_metrics(dataset_transf_test, dataset_transf_test_pred, \
                                          self.unprivileged_groups, self.privileged_groups, disp=True)
        if verbose:
            print("Optimal classification threshold (after fairness processing) = %.4f" % best_ultimate_thres)
            # display(pd.DataFrame(metric_test_aft, columns=metric_test_aft.keys(), index=[0]))
            
        return metric_test_aft

    def predict(self, dataset_orig_test, verbose=False):
        assert self.classifier, 'There is no model to use. Please fit the model first.'
        
        best_ultimate_thres = self.best_ultimate_thres

        # Transform into standardized dataframe
        if not self.is_valid:
            dataset_orig_test[self.label_name] = self.favorable_classes[0]
            dataset_transf_test = generate_formatted_dataframe(dataset_orig_test, label_name=self.label_name, \
                                                            favorable_classes=self.favorable_classes, \
                                                            protected_attribute_names=self.protected_attribute_names, \
                                                            privileged_classes=self.privileged_classes,\
                                                            categorical_features=self.categorical_features, \
                                                            features_to_keep=self.features_to_keep, \
                                                            features_to_drop=self.features_to_drop,\
                                                            na_values=self.na_values, \
                                                            custom_preprocessing=self.custom_preprocessing, \
                                                            metadata=self.metadata)
            print("Data has been transformed into standardized dataframe.")
        else:
            dataset_transf_test = dataset_orig_test.copy(deepcopy=True)
            
        X_test = dataset_transf_test.features
        y_test = dataset_transf_test.labels
        
        # Predict_proba on test data
        pos_ind = dataset_transf_test.favorable_label
        y_pred = self.classifier.predict_proba(X_test)[:,int(pos_ind)].reshape(-1,1)

        fav_inds = y_pred > best_ultimate_thres
        dataset_transf_test.labels[fav_inds] = dataset_transf_test.favorable_label
        dataset_transf_test.labels[~fav_inds] = dataset_transf_test.unfavorable_label

        if verbose:
            print("Optimal classification threshold (after fairness processing) = %.4f" % best_ultimate_thres)

        return dataset_transf_test.labels

