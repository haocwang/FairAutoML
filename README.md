# A Fairness-aware AutoML System
[![GitHub version](https://badge.fury.io/gh/haocwang%2FFairAutoML.svg)](https://badge.fury.io/gh/haocwang%2FFairAutoML) [![PyPI version](https://badge.fury.io/py/FairAutoML.svg)](https://badge.fury.io/py/FairAutoML)

This a ML framework that integrates existing auto-sklearn system with fairness pre-processing and post-processing. Anyone who deals with fairness-sensitive machine learning tasks regardless of their proficiency in machine learning can take advantages of this system.

### Prerequisites

What things you need to install the software and how to install them

```bash
pip install -r requirements.txt
```

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install FairAutoML

```bash
$ pip install FairAutoML
```

## Usage
### Initialize the model
```python
fmodel = AutoML.FairAutoML(input_columns, label_name, favorable_classes, 
                 protected_attribute_names, privileged_classes,
                 privileged_groups, unprivileged_groups,
                 categorical_features, features_to_keep, 
                 features_to_drop, is_valid)
```

### Fit the model
```python
model.fit(train, dataset_metric='mean_difference', dataset_metric_threshold=0.001,
            classifier_metric='Equal opportunity difference', optim_options=None,
            time_left_for_this_task=200, per_run_time_limit=20, train_split_size=0.8,
            verbose=True)
```            

### Predict on test data
```python
pred_labels = model.predict(test)
```

For more detailed example, please check the [demo.ipynb](demo.ipynb).

## Built With
* [auto-sklearn](https://automl.github.io/auto-sklearn/master/) - An automated machine learning toolkit
* [AIF360](https://github.com/IBM/AIF360) - A comprehensive set of fairness metrics for datasets and machine learning models

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors
* **Zihao Guo**
* **Yichao Shen**
* **Nimi Wang**
* **Haochen Wang**

## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2019

