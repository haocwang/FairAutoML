#from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.disparate_impact_remover import DisparateImpactRemover
from aif360.algorithms.preprocessing.lfr import LFR
import warnings
warnings.filterwarnings("ignore")

# preprocessing functions

######################################################################################
## Unable to pass along instance_weights to autosklearn model, so reweighting is abandoned for now.

# def reweighting(dataset, unprivileged_groups, privileged_groups):
#     # train and transform original training with preprocessing reweighting algorithm
#     RW = Reweighing(unprivileged_groups, privileged_groups)
#     RW.fit(dataset)
#     dataset_transf = RW.transform(dataset)
#     return dataset_transf
######################################################################################

def optimPreproc(dataset, optim_options):
    OP = OptimPreproc(OptTools, optim_options)
    OP = OP.fit(dataset)
    # Transform training data and align features
    dataset_transf = OP.transform(dataset, transform_Y=True)
    dataset_transf = dataset.align_datasets(dataset_transf)
    return dataset_transf

def disparateImpactRemover(dataset, repair_level = 1):
    # Use disparate impact remover to generate data
    # repair_level in [0,1]
    DIR = DisparateImpactRemover(repair_level)
    dataset_transf = DIR.fit_transform(dataset)
    return dataset_transf
    
def learningFairRepresentation(dataset, unprivileged_groups, privileged_groups, threshold = 0.5, **kwargs):
    # Use **kwargs for LFR init.
    '''
    k (int, optional): Number of prototypes.
    Ax (float, optional): Input recontruction quality term weight.
    Az (float, optional): Fairness constraint term weight.
    Ay (float, optional): Output prediction error.
    '''
    ## E.x. kwargs = {'k':5, 'Ax': 0.01, 'Ay': 1.0, 'Az': 50.0}
    lFR = LFR(unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups, **kwargs)
    lFR = lFR.fit(dataset)
    dataset = lFR.transform(dataset, threshold)
    dataset
    return dataset
