from sklearn.svm import SVC
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from decereb.chain import Channel, ClassifierDescriptor, FeatureSelectionDescriptor, \
    SchemeDescriptor, Data, ChainBuilder
from sklearn.datasets import make_classification

# Using a slightly more verbose syntax, we can concatenate permutations of analyses using
# (1) different data, (2) different classifiers and (3) different feature selections.

# First we create two different data sets.
X1, y1 = make_classification()
X2, y2 = make_classification()

# Now we setup a list with three different classifiers by using the ClassifierDescriptor class.
# Please note that still all requested permutations for the parameters of each classifier are
# expanded (see simple example).
clf_args_SVC = dict(kernel='linear', C=[[0.1, 1]])
clf_args_RF = dict(criterion=['entropy', 'gini'], n_estimators=128)
clfs = [
    ClassifierDescriptor(name='SVC_rbf', clf=SVC, clf_args=clf_args_SVC),
    ClassifierDescriptor(name='RF', clf=RandomForestClassifier, clf_args=clf_args_RF),
    ClassifierDescriptor(name='LDA', clf=LinearDiscriminantAnalysis)
]

# Dito for different feature selections.
fss = [
    FeatureSelectionDescriptor(name='nested', fs_args=dict(model='nested', threshold='median')),
    FeatureSelectionDescriptor(name='variance', fs_args=dict(threshold=0.01))
]

# A 'channel' specifies a (permutational) processing pipeline consisting of feature selections and
# classifiers.
channel = Channel(fss=fss, clfs=clfs)

# The (processing) scheme descriptor glues together the data input and processing pipeline. The
# scoring argument allows passing of a performance evaluation metric (see sklearn.metrics).
processing_schemes = [
    SchemeDescriptor(name='data_v1', data=Data(X1, y1), channels=channel, scoring='roc_auc'),
    SchemeDescriptor(name='data_v2', data=Data(X2, y2), channels=channel, scoring='roc_auc')
]

# Build the chain, i.e. all permutations of analyses.
analysis = ChainBuilder(schemes=processing_schemes).build_chain()

# And run..
# In this example the chain contains 16 different analyses ('links').
result = analysis.run(n_jobs_links=1, n_jobs_folds=1, verbose=1, output_path='/tmp/decereb/',
                      skip_ioerror=False, skip_runerror=False, detailed_save=True)
print('Finished example!')