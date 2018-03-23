import gc

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, log_loss, precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report

from preprocessing import transform_data, prepare_data, ALL
from plot import plot_roc_curve, plot_precision_recall


def _train_single_file(clf, stats, train_file_name, partial=False, **kwargs):
    print 'Training file ' + train_file_name
    train_file = pd.read_csv(train_file_name).iloc[:, 1:]

    data_set = prepare_data(df=train_file,
                            target_column=kwargs.get('target_column'),
                            id_column=kwargs.get('id_column'),
                            usage=kwargs.get('usage', ALL),
                            columns=kwargs.get('columns'),
                            columns_order=kwargs.get('columns_order'))
    del train_file
    gc.collect()
    data_tr = transform_data(data_set=data_set,
                             stats=stats,
                             integer_columns_slice=kwargs['integer_columns_slice'],
                             category_columns_slice=kwargs['category_columns_slice'])
    del data_set
    gc.collect()

    if partial:
        clf.partial_fit(data_tr.data, data_tr.target, classes=data_tr.target_names)
    else:
        clf.fit(data_tr.data, data_tr.target)

    # Force garbage collection
    del data_tr
    gc.collect()

    return clf


def train_files(clf, stats, num_files, train_file_prefix='train-', **kwargs):
    file_indices = xrange(*num_files)
    if len(file_indices) == 1:
        train_file_name = 'big_data/{}{}.csv'.format(train_file_prefix, 0)
        return _train_single_file(clf, stats, train_file_name, **kwargs)

    for i in file_indices:
        train_file_name = 'big_data/{}{}.csv'.format(train_file_prefix, str(i))
        clf = _train_single_file(clf, stats, train_file_name, partial=True, **kwargs)

    return clf


def _cv_single_file(clf, stats, cv_file_name, **kwargs):
    print 'Cross validating file ' + cv_file_name
    cv_file = pd.read_csv(cv_file_name).iloc[:, 1:]

    data_set = prepare_data(df=cv_file,
                            target_column=kwargs.get('target_column'),
                            id_column=kwargs.get('id_column'),
                            usage=kwargs.get('usage', ALL),
                            columns=kwargs.get('columns'),
                            columns_order=kwargs.get('columns_order'))
    del cv_file
    gc.collect()
    data_tr = transform_data(data_set=data_set,
                             stats=stats,
                             integer_columns_slice=kwargs['integer_columns_slice'],
                             category_columns_slice=kwargs['category_columns_slice'])
    del data_set
    gc.collect()

    # predict
    target_proba = clf.predict_proba(data_tr.data)
    target = clf.predict(data_tr.data)

    a = target_proba[: ,1]
    report = classification_report(data_tr.target, target_proba[:, 1] > 0.63, target_names=data_tr.target_names.astype('|S10'))
    print report

    return data_tr.target, target, target_proba


def cv_files(clf, stats, num_files, cv_file_prefix='train-', **kwargs):
    file_indices = xrange(*num_files)
    if len(file_indices) == 1:
        cv_file_name = 'big_data/{}{}.csv'.format(cv_file_prefix, 0)
        return [_cv_single_file(clf, stats, cv_file_name, **kwargs)]

    cv_values = []
    for i in xrange(*num_files):
        cv_file_name = 'big_data/{}{}.csv'.format(cv_file_prefix, str(i))
        cv_values.append(_cv_single_file(clf, stats, cv_file_name, **kwargs))
    return cv_values


def calculate_precision_recall(scores):
    true_targets = []
    targets = []
    roc_scores = []
    precision_recall_roc_scores = []
    fps = []
    tps = []
    precisions = []
    recalls = []
    thresholds = []
    for true_target, target, target_proba in scores:

        true_targets.extend(true_target)
        targets.extend(target)

        # false and true positives
        fp, tp, thresholds_p = roc_curve(true_target, target)
        # Receiver operating characteristic
        roc_auc = auc(fp, tp)
        precision, recall, pr_thresholds = precision_recall_curve(true_target, target)
        precision_recall_roc_scores.append(auc(recall, precision))

        roc_scores.append(roc_auc)
        fps.append(fp)
        tps.append(tp)
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)

        print 'FP: {}, TP: {}, thresholds: {}, auc: {:.3f}'.format(fp, tp, thresholds_p, roc_auc)
        print 'precision: {}, recall: {}, thresholds: {}'.format(precision, recall, pr_thresholds)

        #plot_roc_curve(fp, tp, roc_auc)

    fp, tp, thresholds_p = roc_curve(true_targets, targets)
    roc_auc = auc(fp, tp)
    precision, recall, pr_thresholds = precision_recall_curve(true_targets, targets)

    # get medium clone
    scores_to_sort = roc_scores
    medium = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
    print("Medium clone is #%i" % medium)

    plot_precision_recall(precision_recall_roc_scores[medium], 'ok', precisions[medium],
                          recalls[medium], "test")

    print 'FP: {}, TP: {}, thresholds: {}, auc: {:.3f}'.format(fp, tp, thresholds_p, roc_auc)
    print 'precision: {}, recall: {}, thresholds: {}'.format(precision, recall, pr_thresholds)

    summary = '%s %s %s %s' % (np.mean(roc_scores), np.std(roc_scores),
                                        np.mean(precision_recall_roc_scores),
                                        np.std(precision_recall_roc_scores))
    print(summary)
    m_precisions = precisions[medium]
    m_recalls = recalls[medium]
    m_thresholds = np.hstack(([0], thresholds[medium]))
    idx80 = m_precisions >= 0.8
    print("P=%.2f R=%.2f thresh=%.2f" % (m_precisions[idx80][0], m_recalls[idx80][0],
                                         m_thresholds[idx80][0]))
    plot_roc_curve(fp, tp, roc_auc)


def calculate_scores(scores, is_log=True):
    # scores
    cv_errors, cv_proba_errors, cv_log_loss_errors = ([] for i in range(3))

    def _set_scores(cv_error, cv_proba_error, cv_log_loss_error):
        (cv_errors.append(cv_error), cv_proba_errors.append(cv_proba_error),
         cv_log_loss_errors.append(cv_log_loss_error))

    for true_target, target, target_proba in scores:
        cv_error = accuracy_score(true_target, target)
        cv_proba_error = accuracy_score(true_target, target_proba.argmax(axis=1))
        cv_log_loss_error = log_loss(true_target, target_proba) if is_log else 0
        _set_scores(cv_error, cv_proba_error, cv_log_loss_error)

        print 'CV Error: {:.5f}'.format(cv_error)
        print 'CV Proba Error: {:.5f}'.format(cv_proba_error)
        print 'CV Log Loss: {:.5f}'.format(cv_log_loss_error)

    print 'CV Errors: [ Mean={:.5f}, Stddev={:.5f} ]'.format(np.mean(cv_errors), np.std(cv_errors))
    print 'CV Proba Errors: [ Mean={:.5f}, Stddev={:.5f} ]'.format(np.mean(cv_proba_errors), np.std(cv_proba_errors))
    print 'CV Log Loss: [ Mean={:.5f}, Stddev={:.5f} ]'.format(np.mean(cv_log_loss_errors), np.std(cv_log_loss_errors))


def predict_files(clf, stats, num_files, predict_file_prefix='test-',
                  result_file_name='test-predict', **kwargs):
    with open('big_data/{}.csv'.format(result_file_name), 'wb+') as f:
        f.write('Id,Predicted\n')
        for i in xrange(*num_files):
            predict_file_name = 'big_data/{}{}.csv'.format(predict_file_prefix, str(i))
            print 'Predicting file ' + predict_file_name
            predict_file = pd.read_csv(predict_file_name).iloc[:, 1:]

            data_set = prepare_data(df=predict_file,
                                    id_column=kwargs.get('id_column'),
                                    usage=kwargs.get('usage', ALL),
                                    columns=kwargs.get('columns'),
                                    columns_order=kwargs.get('columns_order'))
            del predict_file
            gc.collect()
            data_tr = transform_data(data_set=data_set,
                                     stats=stats,
                                     integer_columns_slice=kwargs['integer_columns_slice'],
                                     category_columns_slice=kwargs['category_columns_slice'])
            del data_set
            gc.collect()

            target_predict = clf.predict_proba(data_tr.data)

            # Probability of a click
            target_prob = target_predict[:, 1]
            y_out = np.vstack([data_tr.ids, target_prob]).transpose()

            np.savetxt(f, y_out, delimiter=',', fmt=['%d', '%.4f'])

            # Garbage collection
            del data_tr
            gc.collect()
