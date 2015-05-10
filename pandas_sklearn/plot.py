import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot as pl

from preprocessing import get_feature_location


def plot_3D(data_set, estimation, x, y, z, hide_ticks=False, show_class_names=False):
    x_loc = get_feature_location(data_set, x)
    y_loc = get_feature_location(data_set, y)
    z_loc = get_feature_location(data_set, z)

    data = data_set.data
    target = data_set.target if data_set.has_target else estimation.labels_
    target_unique = np.unique(target)
    target_names = data_set.target_names if data_set.has_target else target_unique
    colors = pl.cm.RdYlBu(np.linspace(0, 1, target_unique.size))

    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')

    for name, label, color in [(target_names[i], i, colors[i]) for i in target_unique]:
        observations = data[:, x_loc][target == label]
        num_observations = observations.size
        label_name = 'class {}, num={}'.format(name, num_observations)
        ax.scatter(data[:, x_loc][target == label],
                   data[:, y_loc][target == label],
                   data[:, z_loc][target == label],
                   s=40,
                   alpha=0.2,
                   color=color)
        if show_class_names:
            ax.text3D(data[target == label, x_loc].mean(),
                      data[target == label, y_loc].mean() + 1.5,
                      data[target == label, z_loc].mean(), label_name,
                      horizontalalignment='center',
                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

    # Reorder the labels to have colors matching the cluster results
    target = np.choose(target, target_unique.tolist()).astype(np.float)
    ax.scatter(data[:, x_loc], data[:, y_loc], data[:, z_loc], c=target, cmap=pl.cm.spectral)

    ax.set_xlim([data[:, x_loc].min(), data[:, x_loc].max()])
    ax.set_ylim([data[:, y_loc].min(), data[:, y_loc].max()])
    ax.set_zlim([data[:, z_loc].min(), data[:, z_loc].max()])

    if hide_ticks:
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    pl.show()


def plot_histograms(data_set, estimation, x, show_legend=True):
    x_loc = get_feature_location(data_set, x)

    data = data_set.data
    target = data_set.target if data_set.has_target else estimation.labels_
    target_unique = np.unique(target)
    target_names = data_set.target_names if data_set.has_target else target_unique
    colors = pl.cm.RdYlBu(np.linspace(0, 1, target_unique.size))

    pl.figure()

    # bin width of the histogram in steps of 20
    min_hist = np.floor(data[:, x_loc].min())
    max_hist = np.ceil(data[:, x_loc].max())
    bins = np.arange(min_hist, max_hist, (max_hist - min_hist) / 50)

    # get the max count for a particular bin for all classes combined
    max_bin = max(np.histogram(data[:, x_loc], bins=bins)[0])

    for name, label, color in [(target_names[i], i, colors[i]) for i in target_unique]:
        observations = data[:, x_loc][target == label]
        mean = np.mean(observations)
        stdev = np.std(observations)
        num_observations = observations.size
        label_name = 'class {} ($\mu={:.2f}$, $\sigma={:.2f}$, $num={}$)'.format(name, mean, stdev,
                                                                               num_observations)
        pl.hist(data[:, x_loc][target == label],
                bins=bins,
                alpha=0.3,
                label=label_name,
                color=color)

    pl.ylim([0, max_bin * 1.5])
    pl.title('')
    pl.xlabel(x)
    pl.ylabel('count')
    if show_legend:
        pl.legend(loc='upper right')
    pl.show()


def plot_scatter(data_set, estimation, x, y, show_legend=True):
    x_loc = get_feature_location(data_set, x)
    y_loc = get_feature_location(data_set, y)

    data = data_set.data
    target = data_set.target if data_set.has_target else estimation.labels_
    target_unique = np.unique(target)
    target_names = data_set.target_names if data_set.has_target else target_unique
    colors = pl.cm.RdYlBu(np.linspace(0, 1, target_unique.size))

    pl.figure()

    for name, label, color in [(target_names[i], i, colors[i]) for i in target_unique]:
        observations = {x: data[:, x_loc][target == label], y: data[:, y_loc][target == label]}
        correlation = pearsonr(observations[x], observations[y])
        num_observations = observations[x].size
        label_name = 'class {}, $R={:.2f}$, $num={}$'.format(name, correlation[0], num_observations)

        pl.scatter(x=observations[x],
                   y=observations[y],
                   color=color,
                   alpha=0.7,
                   label=label_name)

    pl.title('')
    pl.xlabel(x)
    pl.ylabel(y)
    if show_legend:
        pl.legend(loc='upper right')
    pl.show()


def plot_scatter_classes(data_set, estimation, x, show_legend=True):
    x_loc = get_feature_location(data_set, x)

    data = data_set.data
    target = data_set.target if data_set.has_target else estimation.labels_
    target_unique = np.unique(target)
    target_names = data_set.target_names if data_set.has_target else target_unique
    colors = pl.cm.RdYlBu(np.linspace(0, 1, target_unique.size))

    pl.figure()

    for name, label, color in [(target_names[i], i, colors[i]) for i in target_unique]:
        observations = {x: data[:, x_loc][target == label]}
        num_observations = observations[x].size
        label_name = 'class {}, $num={}$'.format(name, num_observations)

        pl.scatter(x=observations[x],
                   y=target[target == label],
                   color=color,
                   alpha=0.7,
                   label=label_name)

    pl.title('')
    pl.xlabel(x)
    pl.ylabel('target')
    if show_legend:
        pl.legend(loc='upper right')
    pl.show()


def plot_roc_curve(fp, tp, roc_auc):
    pl.figure()
    pl.plot(fp, tp, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()


def plot_precision_recall(auc_score, name, precision, recall, label=None):
    pl.figure(num=None, figsize=(6, 5))
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    pl.fill_between(recall, precision, alpha=0.5)
    pl.grid(True, linestyle='-', color='0.75')
    pl.plot(recall, precision, lw=1)
