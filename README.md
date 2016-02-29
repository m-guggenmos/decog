# decereb
High-level interface for sklearn, targeted at neuroimaging data analysis. Not even alpha at the moment -- the API expected to change on a daily basis.

Example:

```python
from sklearn.svm import SVC
from decereb.chain import SimpleChain
from sklearn.datasets import make_classification

X, y = make_classification()
clf = SVC
# Decereb offers two features for classifier and feature selection parameters:
# 1. Parameters in plain list form [Param1, Param2, .. ParamN] generate N analyses looping over
#    each parameter.
#    In the example below, two analyses are produced, one using an RBF kernel and a second one
#    using a linear kernel.
# 2. Parameters defined as a nested list [[Param1, Param2, .. ParamN]] are iterated over in a
#    nested leave-one-out cross-validation procedure using GridSearchCV, to find the optimal
#    parameter in each fold.
#    In the example below, a gridsearch is performed for the parameter C on the values 0.1 and 1,
#    both for the analysis using an RBF kernel and for the analysis using a linear kernel.
clf_args = dict(kernel=['rbf', 'linear'], C=[[0.1, 1]])

# Here we use a variance-based feature selection and start two analyses with two different
# thresholds. Please note that Decereb expands all permutations of classifier parameters and feature
# selection parameters, i.e. in this case the code produces 2x2=4 analyses.
fs = 'variance'
fs_args = dict(threshold=[0.01, 0.1])

analysis = SimpleChain(data=X, clf=clf, clf_args=clf_args, fs=fs, fs_args=fs_args, labels=y)


# n_jobs_links:  distribute different analyses across n_jobs_links processors
# n_jobs_folds:  distribute cross-validation folds of each analysis across n_jobs_folds processors
# verbose:       increase in order to view more detailed results (e.g. fold-wise results)
# output_path:   directory to store the result
# skip_ioerror:  set True if the analysis chain should continue despite an I/O error when saving the,
#                previous result
# skip_runerror: set True if the analysis chain should continue despite any runtime error in the
#                previous analysis
# detailed_save: save more detailed results (currently undocumented)


result = analysis.run(n_jobs_links=1, n_jobs_folds=1, verbose=1, output_path='/tmp/decereb/',
                      skip_ioerror=False, skip_runerror=False, detailed_save=True)

print('Finished example!')

# Decereb results are stored in a simple SQL-based database (https://dataset.readthedocs.org/),
# which allows easy querying of the results (e.g. SELECT * FROM chain ORDER BY roc_auc_scores)
#
# An exception is if the classifier is searchlight-based, in which case the results are saved as
# nibabel.NiftiImages.
```
