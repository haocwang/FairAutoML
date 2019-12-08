from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.postprocessing import EqOddsPostprocessing, RejectOptionClassification

def eqOddsPostprocessing(dataset, unprivileged_groups, privileged_groups):
    eqOdds = EqOddsPostprocessing(unprivileged_groups, privileged_groups)
    dataset = eqOdds.fit_transform(dataset)
    return dataset

def rejectOptionClassification(dataset, unprivileged_groups, privileged_groups):
    rejectOption = RejectOptionClassification(unprivileged_groups, privileged_groups)
    dataset = rejectOption.fit_transform(dataset)
    return dataset
