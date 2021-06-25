from collections import defaultdict
from contextlib import suppress
from functools import partial
from itertools import chain, product
from traceback import format_exc
from turtledemo.chaos import g
import time

import numpy as np
from abc import ABCMeta, abstractmethod

from joblib import Parallel, delayed, logger
from sklearn.base import is_classifier, clone, _is_pairwise
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer, _PredictScorer, _cached_call
from sklearn.model_selection._search import BaseSearchCV, _check_param_grid, ParameterGrid
from sklearn.model_selection._split import BaseShuffleSplit, check_cv, _BaseKFold, _validate_shuffle_split
from sklearn.model_selection._validation import  _insert_error_scores, _score
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _deprecate_positional_args, _check_fit_params
from sklearn.utils.validation import _num_samples, column_or_1d
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target
import warnings
from sklearn.utils import indexable, check_random_state, _safe_indexing, _approximate_mode
from math import floor, ceil
from sklearn.model_selection import train_test_split
import numbers
from sklearn.model_selection import GroupKFold, StratifiedKFold, BaseCrossValidator

# don't import the reimplemented base classes.
__all__ = ['StratifiedUserKFold','UserStratifiedShuffleSplit','train_test_user_stratified_split']

class StratifiedUserKFold(_BaseKFold):

    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        if y is None:
            raise ValueError("The 'y' parameter should not be None.")
        y = check_array(y,ensure_2d=False,dtype=None)


        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups)

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]
        #print(f'n_samples_per_group: {n_samples_per_group}')
        #print(f'indices: {indices}')

        #POSSIBLE CHANGE:
        #should be able to set n_samples_per_group to np.ones(len(indices)) to weigh all users as weight 1
        # this may work better or worse, but might be worth asking Dr. Ruozzi about...
        n_samples_per_group = np.ones(len(n_samples_per_group))

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Use fresh weights for each fold, for each class to ensure balancing among classes too.
        for yval in np.unique(y):

            # Total weight of each fold
            n_samples_per_fold = np.zeros(self.n_splits)

            # Distribute samples by adding the largest weight to the lightest fold
            for group_index, weight in enumerate(n_samples_per_group):
                #print(y[np.where(groups==group_index)[0]])
                if y[np.where(groups==indices[group_index])[0]][0] != yval:
                    continue
                lightest_fold = np.argmin(n_samples_per_fold)
                #print(f'sorting {indices[group_index]} ({yval}) to {lightest_fold}')
                n_samples_per_fold[lightest_fold] += weight
                group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

   #
   # def _make_test_folds(self, X, y, groups):
   #     rng = check_random_state(self.random_state)
   #     y = np.asarray(y)
   #     type_of_target_y = type_of_target(y)
   #     allowed_target_types = ('binary', 'multiclass')
   #     if type_of_target_y not in allowed_target_types:
   #         raise ValueError(
   #             'Supported target types are: {}. Got {!r} instead.'.format(
   #                 allowed_target_types, type_of_target_y))

   #     y = column_or_1d(y)

   #     _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
   #     # y_inv encodes y according to lexicographic order. We invert y_idx to
   #     # map the classes so that they are encoded by order of appearance:
   #     # 0 represents the first label appearing in y, 1 the second, etc.
   #     _, class_perm = np.unique(y_idx, return_inverse=True)
   #     y_encoded = class_perm[y_inv]

   #     n_classes = len(y_idx)
   #     y_counts = np.bincount(y_encoded)
   #     min_groups = np.min(y_counts)
   #     if np.all(self.n_splits > y_counts):
   #         raise ValueError("n_splits=%d cannot be greater than the"
   #                          " number of members in each class."
   #                          % (self.n_splits))
   #     if self.n_splits > min_groups:
   #         warnings.warn(("The least populated class in y has only %d"
   #                        " members, which is less than n_splits=%d."
   #                        % (min_groups, self.n_splits)), UserWarning)

   #     # Determine the optimal number of samples from each class in each fold,
   #     # using round robin over the sorted y. (This can be done direct from
   #     # counts, but that code is unreadable.)
   #     y_order = np.sort(y_encoded)
   #     allocation = np.asarray(
   #         [np.bincount(y_order[i::self.n_splits], minlength=n_classes)
   #          for i in range(self.n_splits)])

   #     # To maintain the data order dependencies as best as possible within
   #     # the stratification constraint, we assign samples from each class in
   #     # blocks (and then mess that up when shuffle=True).
   #     test_folds = np.empty(len(y), dtype='i')
   #     for k in range(n_classes):
   #         # since the kth column of allocation stores the number of samples
   #         # of class k in each test set, this generates blocks of fold
   #         # indices corresponding to the allocation for class k.
   #         folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
   #         if self.shuffle:
   #             rng.shuffle(folds_for_class)
   #         test_folds[y_encoded == k] = folds_for_class
   #     return test_folds

   # def _iter_test_masks(self, X, y=None, groups=None):
   #     test_folds = self._make_test_folds(X, y, groups)
   #     for i in range(self.n_splits):
   #         yield test_folds == i


    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y, groups)

class UserStratifiedShuffleSplit(BaseShuffleSplit):

    @_deprecate_positional_args
    def __init__(self, n_splits=10, *, test_size=None, train_size=None,
                 random_state=None):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)
        self._default_test_size = 0.1

    def _iter_indices(self, X, y, groups):

        actual_X = X
        actual_y = y
        actual_groups = groups

        # change everything to look like each participant has one sample and reuse code
        # maintain order within users
        users, user_indices, user_inverse = np.unique(groups,return_index=True,return_inverse=True)
        user_classes = np.zeros(len(users))
        for i,v in enumerate(user_indices):
            user_classes[i] = y[v]

        X = users
        y = user_classes

        n_samples = _num_samples(X)
        y = check_array(y, ensure_2d=False, dtype=None)


        n_train, n_test = _validate_shuffle_split(
            n_samples, self.test_size, self.train_size,
            default_test_size=self._default_test_size)

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([' '.join(row.astype('str')) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)

        if np.min(class_counts) < 2:
            raise ValueError("The least populated class in y has only 1"
                             " member, which is too few. The minimum"
                             " number of groups for any class cannot"
                             " be less than 2.")

        if n_train < n_classes:
            raise ValueError('The train_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_train, n_classes))
        if n_test < n_classes:
            raise ValueError('The test_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_test, n_classes))

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(np.argsort(y_indices, kind='mergesort'),
                                 np.cumsum(class_counts)[:-1])

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)

            train = []
            test = []

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation,
                                                             mode='clip')

                train.extend(perm_indices_class_i[:n_i[i]])
                test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])

            #at this point, train and test are in user-level representation
            # and need to be converted back to vector-level representation.


            train = rng.permutation(train)
            test = rng.permutation(test)

            #print(train,test)

            #want to return indices of the train and test groups...
            #We can get a list of ids as follows:
            #trainoutindices = [i for i, id in enumerate(actual_groups) if id in train]
            #testoutindices = [i for i, id in enumerate(actual_groups) if id in test]
            #print(f'user inverse of train[1] = {users[train[1]]}')
            #print(f'user_inverse = {user_inverse} ({len(user_inverse)})')
            trainoutindices = np.concatenate([np.where(actual_groups==users[id]) for id in train],axis=1)
            testoutindices = np.concatenate([np.where(actual_groups==users[id]) for id in test],axis=1)

                #break
            #np.where(actual_X==
            yield trainoutindices,testoutindices


            #yield train, test



    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,) or (n_samples, n_labels)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)

    pass

def train_test_user_stratified_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True):
    n_arrays = len(arrays)
    if n_arrays < 3:
        raise ValueError("Need to have at least X, y, groups as input")
    arrays = indexable(*arrays)

    #replace n_samples with the number of users
    n_samples = len(np.unique(arrays[2]))

    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=0.25)

    cv = UserStratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=random_state)
    train, test = next(cv.split(X=arrays[0],y=arrays[1],groups=arrays[2]))
    return list(chain.from_iterable((_safe_indexing(a, train)[0],
                                     _safe_indexing(a, test)[0]) for a in arrays))

# Tell nose that train_test_user_stratified_split is not a test.
setattr(train_test_user_stratified_split, '__test__', False)


class BaseUserLevelSearchCV(BaseSearchCV):
    def score(self, X, y=None, groups=None):
        """Returns the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups: array-like of shape (n_samples, n_output) \
                or (n_samples,), default=None
                Groups to use for plurality scoring;
                None for if it's not used in the scoring method.

        Returns
        -------
        score : float
        """
        self._check_is_fitted('score')
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y, groups)

        # callable
        #print(self.scorer_)
        score = self.scorer_(self.best_estimator_, X, y, groups)
        if self.multimetric_:
            score = score[self.refit]
        return score

    def _group_fit_and_score(estimator, X, y, groups, scorer, train, test, verbose,
                       parameters, fit_params, return_train_score=False,
                       return_parameters=False, return_n_test_samples=False,
                       return_times=False, return_estimator=False,
                       split_progress=None, candidate_progress=None,
                       error_score=np.nan):

        """Fit estimator and compute scores for a given dataset split.

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        X : array-like of shape (n_samples, n_features)
            The data to fit.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
            The target variable to try to predict in the case of
            supervised learning.

        scorer : A single callable or dict mapping scorer name to the callable
            If it is a single callable, the return value for ``train_scores`` and
            ``test_scores`` is a single float.

            For a dict, it should be one mapping the scorer name to the scorer
            callable object / function.

            The callable object / fn should have signature
            ``scorer(estimator, X, y)``.

        train : array-like of shape (n_train_samples,)
            Indices of training samples.

        test : array-like of shape (n_test_samples,)
            Indices of test samples.

        verbose : int
            The verbosity level.

        error_score : 'raise' or numeric, default=np.nan
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised.
            If a numeric value is given, FitFailedWarning is raised.

        parameters : dict or None
            Parameters to be set on the estimator.

        fit_params : dict or None
            Parameters that will be passed to ``estimator.fit``.

        groups: array-like of shape (n_samples,) or (n_samples, n_outputs) or None
            The groups to pass to the scoring function.

        return_train_score : bool, default=False
            Compute and return score on training set.

        return_parameters : bool, default=False
            Return parameters that has been used for the estimator.

        split_progress : {list, tuple} of int, default=None
            A list or tuple of format (<current_split_id>, <total_num_of_splits>).

        candidate_progress : {list, tuple} of int, default=None
            A list or tuple of format
            (<current_candidate_id>, <total_number_of_candidates>).

        return_n_test_samples : bool, default=False
            Whether to return the ``n_test_samples``.

        return_times : bool, default=False
            Whether to return the fit/score times.

        return_estimator : bool, default=False
            Whether to return the fitted estimator.

        Returns
        -------
        result : dict with the following attributes
            train_scores : dict of scorer name -> float
                Score on training set (for all the scorers),
                returned only if `return_train_score` is `True`.
            test_scores : dict of scorer name -> float
                Score on testing set (for all the scorers).
            n_test_samples : int
                Number of test samples.
            fit_time : float
                Time spent for fitting in seconds.
            score_time : float
                Time spent for scoring in seconds.
            parameters : dict or None
                The parameters that have been evaluated.
            estimator : estimator object
                The fitted estimator.
            fit_failed : bool
                The estimator failed to fit.
        """
        if not isinstance(error_score, numbers.Number) and error_score != 'raise':
            raise ValueError(
                "error_score must be the string 'raise' or a numeric value. "
                "(Hint: if using 'raise', please make sure that it has been "
                "spelled correctly.)"
            )

        progress_msg = ""
        if verbose > 2:
            if split_progress is not None:
                progress_msg = f" {split_progress[0] + 1}/{split_progress[1]}"
            if candidate_progress and verbose > 9:
                progress_msg += (f"; {candidate_progress[0] + 1}/"
                                 f"{candidate_progress[1]}")

        if verbose > 1:
            if parameters is None:
                params_msg = ''
            else:
                sorted_keys = sorted(parameters)  # Ensure deterministic o/p
                params_msg = (', '.join(f'{k}={parameters[k]}'
                                        for k in sorted_keys))
        if verbose > 9:
            start_msg = f"[CV{progress_msg}] START {params_msg}"
            print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

        # Adjust length of sample weights
        fit_params = fit_params if fit_params is not None else {}
        fit_params = _check_fit_params(X, fit_params, train)

        if parameters is not None:
            # clone after setting parameters in case any parameters
            # are estimators (like pipeline steps)
            # because pipeline doesn't clone steps in fit
            cloned_parameters = {}
            for k, v in parameters.items():
                cloned_parameters[k] = clone(v, safe=False)

            estimator = estimator.set_params(**cloned_parameters)

        start_time = time.time()

        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)

        result = {}
        try:
            if y_train is None:
                estimator.fit(X_train, **fit_params)
            else:
                estimator.fit(X_train, y_train, **fit_params)

        except Exception as e:
            # Note fit time as time until error
            fit_time = time.time() - start_time
            score_time = 0.0
            if error_score == 'raise':
                raise
            elif isinstance(error_score, numbers.Number):
                if isinstance(scorer, dict):
                    test_scores = {name: error_score for name in scorer}
                    if return_train_score:
                        train_scores = test_scores.copy()
                else:
                    test_scores = error_score
                    if return_train_score:
                        train_scores = error_score
                warnings.warn("Estimator fit failed. The score on this train-test"
                              " partition for these parameters will be set to %f. "
                              "Details: \n%s" %
                              (error_score, format_exc()),
                              FitFailedWarning)
            result["fit_failed"] = True
        else:
            result["fit_failed"] = False

            fit_time = time.time() - start_time
            test_scores = _score(estimator, X_test, y_test, scorer, error_score)
            score_time = time.time() - start_time - fit_time
            if return_train_score:
                train_scores = _score(
                    estimator, X_train, y_train, scorer, error_score
                )

        if verbose > 1:
            total_time = score_time + fit_time
            end_msg = f"[CV{progress_msg}] END "
            result_msg = params_msg + (";" if params_msg else "")
            if verbose > 2 and isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            result_msg += f" total time={logger.short_format_time(total_time)}"

            # Right align the result_msg
            end_msg += "." * (80 - len(end_msg) - len(result_msg))
            end_msg += result_msg
            print(end_msg)

        result["test_scores"] = test_scores
        if return_train_score:
            result["train_scores"] = train_scores
        if return_n_test_samples:
            result["n_test_samples"] = _num_samples(X_test)
        if return_times:
            result["fit_time"] = fit_time
            result["score_time"] = score_time
        if return_parameters:
            result["parameters"] = parameters
        if return_estimator:
            result["estimator"] = estimator
        return result

    @_deprecate_positional_args
    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)

        if self.verbose > 9:
            print(f'in fit, X,y,groups shapes are: {X.shape}, {y.shape}, {groups.shape}')
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs,
                            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None,
                                    more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(delayed(_group_fit_and_score)(clone(base_estimator),
                                                       X, y, groups,
                                                       train=train, test=test,
                                                       parameters=parameters,
                                                       split_progress=(
                                                           split_idx,
                                                           n_splits),
                                                       candidate_progress=(
                                                           cand_idx,
                                                           n_candidates),
                                                       **fit_and_score_kwargs)
                               for (cand_idx, parameters),
                                   (split_idx, (train, test)) in product(
                    enumerate(candidate_params),
                    enumerate(cv.split(X, y, groups))))

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out,
                    all_more_results)

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]['test_scores']
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                        self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                    self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

class GridUserLevelSearchCV(BaseUserLevelSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method.
    It also implements "score_samples", "predict", "predict_proba",
    "decision_function", "transform" and "inverse_transform" if they are
    implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : str, callable, list/tuple or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    pre_dispatch : int, or str, default=n_jobs
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        .. versionchanged:: 0.20
            Support for callable added.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``


    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = svm.SVC()
    >>> clf = GridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    >>> sorted(clf.cv_results_.keys())
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split2_test_score', ...
     'std_fit_time', 'std_score_time', 'std_test_score']

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

        .. versionadded:: 0.20

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    ParameterGrid : Generates all the combinations of a hyperparameter grid.
    train_test_split : Utility function to split the data into a development
        set usable for fitting a GridSearchCV instance and an evaluation set
        for its final evaluation.
    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.

    """
    _required_parameters = ["estimator", "param_grid"]

    @_deprecate_positional_args
    def __init__(self, estimator, param_grid, *, scoring=None,
                 n_jobs=None, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))


def _group_fit_and_score(estimator, X, y, groups, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, return_estimator=False,
                   split_progress=None, candidate_progress=None,
                   error_score=np.nan):

    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : bool, default=False
        Compute and return score on training set.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_failed : bool
            The estimator failed to fit.
    """
    if not isinstance(error_score, numbers.Number) and error_score != 'raise':
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += (f"; {candidate_progress[0]+1}/"
                             f"{candidate_progress[1]}")

    if verbose > 1:
        if parameters is None:
            params_msg = ''
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = (', '.join(f'{k}={parameters[k]}'
                                    for k in sorted_keys))
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train, groups_train = _group_safe_split(estimator, X, y, groups, train)
    X_test, y_test, groups_test = _group_safe_split(estimator, X, y, groups, test, train)


    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exc()),
                          FitFailedWarning)
        result["fit_failed"] = True
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time
        test_scores = _group_score(estimator, X_test, y_test, groups_test, scorer, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _group_score(
                estimator, X_train, y_train, groups_test, scorer, error_score
            )

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2 and isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                result_msg += f" {scorer_name}: ("
                if return_train_score:
                    scorer_scores = train_scores[scorer_name]
                    result_msg += f"train={scorer_scores:.3f}, "
                result_msg += f"test={test_scores[scorer_name]:.3f})"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def _group_score(estimator, X_test, y_test, group_test, scorer, error_score="raise"):
    """Compute the score(s) of an estimator on a given test set.

    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    if isinstance(scorer, dict):
        # will cache method calls if needed. scorer() returns a dict
        scorer = _MultimetricScorer(**scorer)

    try:
        if y_test is None:
            scores = scorer(estimator, X_test)
        elif group_test is None:# or scorer.__getattribute__(''):
            scores = scorer(estimator, X_test, y_test)
        else:
            scores = scorer(estimator, X_test, y_test, group_test)
    except Exception:
        if error_score == 'raise':
            raise
        else:
            if isinstance(scorer, _MultimetricScorer):
                scores = {name: error_score for name in scorer._scorers}
            else:
                scores = error_score
            warnings.warn(
                f"Scoring failed. The score on this train-test partition for "
                f"these parameters will be set to {error_score}. Details: \n"
                f"{format_exc()}",
                UserWarning,
            )

    error_msg = (
        "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    )
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, 'item'):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, 'item'):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores



def _group_safe_split(estimator, X, y, groups, indices, train_indices=None):
    if _is_pairwise(estimator):
        if not hasattr(X, "shape"):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]
        else:
            X_subset = X[np.ix_(indices, train_indices)]
    else:
        X_subset = _safe_indexing(X, indices)

    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None

    if groups is not None:
        groups_subset = _safe_indexing(groups, indices)
    else:
        groups_subset = None

    return X_subset, y_subset, groups_subset

class _GroupPredictScorer(_PredictScorer):
    def _score(self, method_caller, estimator, X, y_true, groups, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_pred = method_caller(estimator, "predict", X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred, groups,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, groups,
                                                 **self._kwargs)


def UserLevelScore(scorefunc):
    def returnedFunc(y_true,y_pred,users,*kargs,sample_weight=None):
        y_true, y_pred = y_true.astype(np.int64), y_pred.astype(np.int64)
        user_level_y_true = np.array( [np.bincount(y_true[np.where(users==uid)]).argmax() for uid in np.unique(users)] )
        user_level_y_pred = np.array( [np.bincount(y_pred[np.where(users==uid)]).argmax() for uid in np.unique(users)] )
        return scorefunc(user_level_y_true,user_level_y_pred, *kargs, sample_weight=None)
    return returnedFunc


@_deprecate_positional_args
def make_userlevel_scorer(score_func, *, greater_is_better=True, needs_proba=False,
                needs_threshold=False, **kwargs):
    """Make a scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in
    :class:`~sklearn.model_selection.GridSearchCV` and
    :func:`~sklearn.model_selection.cross_val_score`.
    It takes a score function, such as :func:`~sklearn.metrics.accuracy_score`,
    :func:`~sklearn.metrics.mean_squared_error`,
    :func:`~sklearn.metrics.adjusted_rand_index` or
    :func:`~sklearn.metrics.average_precision`
    and returns a callable that scores an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model to be evaluated, `X` is the data and `y` is the
    ground truth labeling (or `None` in the case of unsupervised models).

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : bool, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    needs_proba : bool, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

    needs_threshold : bool, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)

    Notes
    -----
    If `needs_proba=False` and `needs_threshold=False`, the score
    function is supposed to accept the output of :term:`predict`. If
    `needs_proba=True`, the score function is supposed to accept the
    output of :term:`predict_proba` (For binary `y_true`, the score function is
    supposed to accept probability of the positive class). If
    `needs_threshold=True`, the score function is supposed to accept the
    output of :term:`decision_function`.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True,"
                         " but not both.")
    if needs_proba:
        raise NotImplementedError()
        #cls = _ProbaScorer
    elif needs_threshold:
        raise NotImplementedError()
        #cls = _ThresholdScorer
    else:
        cls = _UserLevelPredictScorer
    return cls(UserLevelScore(score_func), sign, kwargs)


class _BaseUserLevelScorer:
    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign

    @staticmethod
    def _check_pos_label(pos_label, classes):
        if pos_label not in list(classes):
            raise ValueError(
                f"pos_label={pos_label} is not a valid label: {classes}"
            )

    def _select_proba_binary(self, y_pred, classes):
        """Select the column of the positive label in `y_pred` when
        probabilities are provided.

        Parameters
        ----------
        y_pred : ndarray of shape (n_samples, n_classes)
            The prediction given by `predict_proba`.

        classes : ndarray of shape (n_classes,)
            The class labels for the estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Probability predictions of the positive class.
        """
        if y_pred.shape[1] == 2:
            pos_label = self._kwargs.get("pos_label", classes[1])
            self._check_pos_label(pos_label, classes)
            col_idx = np.flatnonzero(classes == pos_label)[0]
            return y_pred[:, col_idx]

        err_msg = (
            f"Got predict_proba of shape {y_pred.shape}, but need "
            f"classifier with two classes for {self._score_func.__name__} "
            f"scoring"
        )
        raise ValueError(err_msg)

    def __repr__(self):
        kwargs_string = "".join([", %s=%s" % (str(k), str(v))
                                 for k, v in self._kwargs.items()])
        return ("make_scorer(%s%s%s%s)"
                % (self._score_func.__name__,
                   "" if self._sign > 0 else ", greater_is_better=False",
                   self._factory_args(), kwargs_string))

    def __call__(self, estimator, X, y_true, users, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        return self._score(partial(_cached_call, None), estimator, X, y_true, users,
                           sample_weight=sample_weight)

    def _factory_args(self):
        """Return non-default make_scorer arguments for repr."""
        return ""



class _UserLevelPredictScorer(_BaseUserLevelScorer):
    def _score(self, method_caller, estimator, X, y_true, users, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_pred = method_caller(estimator, "predict", X)

        #print(f'y_true: {y_true}, y_pred: {y_pred}')
        #return 0.0

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred, users,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, users,
                                                 **self._kwargs)





