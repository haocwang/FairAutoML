from collections import OrderedDict
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
import autosklearn.classification
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def generate_formatted_dataframe(df, label_name, favorable_classes, 
                                    protected_attribute_names, privileged_classes, 
                                    categorical_features, 
                                    features_to_keep,
                                    features_to_drop, 
                                    na_values,
                                    custom_preprocessing,
                                    metadata):
    '''
    @usage:
        to transform the input data into accepted Standard Dataset and split 
        into training, validation (and testing)
        return: 'standardized' splitted dataframe into training, validation 
        (and testing)
    @param:
        - df: original input pandas dataframe
    '''
    # Transform into standardized dataframe
    dataset_standardized = StandardDataset(df, label_name, favorable_classes, 
                                    protected_attribute_names, privileged_classes, 
                                    categorical_features=categorical_features, 
                                    features_to_keep=features_to_keep,
                                    features_to_drop=features_to_drop, 
                                    na_values=na_values,
                                    custom_preprocessing = custom_preprocessing,
                                    metadata = metadata)
    
    return dataset_standardized

def train_test_split(df, train_split_size, shuffle=True):
    '''
    @usage:
        to split dataset into train and test
    @param:
        - df: standardized dataframe
        - train_split_size: float, (0,1)
        - shuffle: bool
    '''
    assert (df.favorable_label and df.label_names), 'The input data is not of standardized form, please use the generate_formatted_dataframe to standardize it first'
    dataset_stand_train, dataset_stand_test = \
        df.split([train_split_size], shuffle=True)
    return dataset_stand_train, dataset_stand_test

def precheck_data_fairness(metric_name, metric_threshold, metric):
    '''
    @usage:
        Check if raw data satisfies the fairness criterion users give.
        Optimally, 'mean_difference' should be larger than -0.1 and less than 0.1, 
        and 'disparate_impact' should be greater than 0.8 and less than 1.2.
        
    @return: flag: True if data is fair. False if data is not fair
    '''
    if metric_name == 'mean_difference' or metric_name == 'statistical_parity_difference':
        metric = metric.mean_difference()
        if metric > 0:
            print('Potential Error: Your priviledged groups might actually be unpriviledged, and vice versa. Please check your (un)priviledged groups.')
        print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric)
        no_preproc = np.abs(metric) < metric_threshold
        print("No discrimination potential, no need for fairness preprocessing: ", no_preproc)
        return no_preproc
    
    elif metric_name == 'disparate_impact':
        metric = metric.disparate_impact()
        if metric > 1:
            print('Potential Error: Your priviledged groups might actually be unpriviledged, and vice versa. Please check your (un)priviledged groups.')
        print("Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances) = %f" % metric)
        no_preproc = (metric > metric_threshold) and (metric < (1.0 + np.abs(1.0 - metric_threshold)))
        print("No discrimination potential, no need for fairness preprocessing: ", no_preproc)
        return no_preproc
    
    elif metric_name == 'consistency':
        metric = metric.consistency()
        print("Consistency of labels = %f" % metric)
        no_preproc = metric > metric_threshold
        print("No discrimination potential, no need for fairness preprocessing: ", no_preproc)
        return no_preproc
    


def postcheck_data_fairness(metric_name, metric, preproc_name):
    # print the fairness metric for transformed train dataset
    if metric_name == 'mean_difference' or metric_name == 'statistical_parity_difference':
        metric = metric.mean_difference()
        print("Difference in mean outcomes between unprivileged and privileged groups after {} = {}".format(preproc_name, metric))
        
    elif metric_name == "disparate_impact":
        metric = metric.disparate_impact()
        print("Disparate impact (probability of favorable outcome for unprivileged instances / probability of favorable outcome for privileged instances) after {} = {}".format(preproc_name, metric))
        
    elif metric_name == 'consistency':
        metric = metric.consistency()
        print("Consistency of labels after {} = {}".format(preproc_name, metric))
    
    
def model_and_find_threshold(train, val, unprivileged_groups, privileged_groups, no_process, classifier_metric="Disparate impact", time_left_for_this_task=30, per_run_time_limit=10):
    try:
        print("--------------Start Fitting Model--------------")
        # 1) fit automl ##(大约20分钟)
        X_train = train.features
        y_train = train.labels
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task, per_run_time_limit)
        automl.fit(X_train, y_train)

        # positive class label
        pos_ind = int(train.favorable_label) ## the good label. just 1

        # obtain val set
        X_valid = val.features
        y_valid = val.labels

        # 2) find optimal threshold by calling predict_proba of the classifier
        ## Find the optimal classification threshold from the **validation set**
        num_thresh = 50
        fairness_metric_arr = []
        ba_arr = np.zeros(num_thresh)
        class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
        for idx, class_thresh in enumerate(class_thresh_arr):

            val_pred = val.copy(deepcopy=True)
            #对于每一行data，我们可以得到negative的概率和positive的概率, 我们把每行data的positive概率放到.scores里面
            val_pred.scores = automl.predict_proba(X_valid)[:,pos_ind].reshape(-1,1) 
            fav_inds = val_pred.scores > class_thresh # fav_inds: bool -- 如果prob大于class_thres, True, fav_lable
            val_pred.labels[fav_inds] = val_pred.favorable_label
            val_pred.labels[~fav_inds] = val_pred.unfavorable_label

            metric_test_bef = compute_metrics(val, val_pred, 
                                          unprivileged_groups, privileged_groups,
                                          disp = False)
            fairness_metric_arr.append(metric_test_bef[classifier_metric])
            ba_arr[idx] = metric_test_bef['Balanced accuracy'] # balanced accuracy

        # 3). Find the classification threshold within the optimal range of fairness
        ## define the fairness metrics optimal range
        metric_range = {"Statistical parity difference":[-0.1,0.1], \
                       "Mean difference": [-0.1, 0.1], \
                       "Disparate impact": [0.8, 1.2], \
                       "Average odds difference": [-0.1,0.1], \
                       "Equal opportunity difference":[-0.1, 0.1], \
                       "Theil index": [0, 0.2]}
        fairness_metric_arr = np.array(fairness_metric_arr)
        lo, hi = metric_range[classifier_metric]

        # Generate the index list where the fairness metrics satisfies the criterion we set
        idx_arr = np.intersect1d(np.argwhere(fairness_metric_arr > lo), np.argwhere(fairness_metric_arr < hi))

        # Find the best threshold and best accuracy
        no_postproc = True
        if idx_arr.size == 0:
            # no optimal range. Do postprocessing
            no_postproc = False
            best_thres = None
            best_acc = 0
        elif idx_arr.size == 1:
            best_thres = class_thresh_arr[idx_arr[0]]
        else:
            best_thres = class_thresh_arr[idx_arr[0]]
            best_acc = ba_arr[idx_arr[0]]
            for idx in idx_arr[1:]:
                if ba_arr[idx] > best_acc:
                    best_thres = class_thresh_arr[idx]
                    best_acc = ba_arr[idx]

        if no_process:
            print("Best balanced accuracy (no fairness processing) considering the fairness = %.4f" % best_acc)
            print("Optimal classification threshold (no fairness processing) = %.4f" % best_thres)
            print()
            return automl, best_thres
        else:
            print("Best balanced accuracy (after fairness processing) = %.4f" % best_acc)
            print("Optimal classification threshold (after fairness processing) = %.4f" % best_thres)
            print()
            return automl, best_thres, best_acc, fairness_metric_arr, ba_arr, no_postproc, lo, hi
    except:
        pass

    
def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true, dataset_pred, 
        unprivileged_groups=unprivileged_groups, 
        privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = \
        0.5*(classified_metric_pred.true_positive_rate()+ 
            classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = \
        classified_metric_pred.statistical_parity_difference()
    metrics["Mean difference"] = \
        classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = \
        classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = \
        classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = \
        classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics
