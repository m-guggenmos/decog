import numpy as np


def decision_trees(clf, drawing_param, nFeatures=None, show_trees=True, show_importance=True, showfig=True):
    from sklearn import tree
    from sklearn.externals.six import StringIO
    import pydot
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    import mgplot.plotting as myplt

    clfs = clf.estimators_ if hasattr(clf, 'estimators_') else [clf]

    if show_trees:
        for i, estimator in enumerate(clfs):
            # create graph
            dot_data = StringIO()
            tree.export_graphviz(estimator, out_file=dot_data, rounded=True, filled=True, **drawing_param)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())

            # create image from graph
            png_str = graph.create_png(prog='dot')
            sio = StringIO()
            sio.write(png_str)
            sio.seek(0)
            img = mpimg.imread(sio)

            # plot the image
            fig = plt.figure('Decision tree %s' % i)
            plt.imshow(img, aspect='equal')
            fig.tight_layout()
            if showfig:
                plt.show()

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for i, f in enumerate(indices):
        print("%d. feature %d = %f [%s]" % (i + 1, f, importances[f], drawing_param['feature_names'][f]))

    if show_importance:
        axis = myplt.hbar([tree.feature_importances_ for tree in clfs], figtitle='Feature importance', nBars=nFeatures,
                          yticknames=drawing_param['feature_names'], xlabel='Gini importance', sort='descend')[0]
        axis.figure.tight_layout()
        if showfig:
            plt.show()
