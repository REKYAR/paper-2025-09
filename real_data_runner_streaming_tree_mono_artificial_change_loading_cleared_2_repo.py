import os
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter, deque
import abc
import numbers
import typing
import math
from tqdm import tqdm
import json
import glob
import warnings
import zipfile
import argparse
from typing import List, Dict, Tuple, Optional, Union, Set, Any
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

try:
    from river import metrics
    from river.tree.base import Leaf
    from river.tree.utils import BranchFactory
    from river.tree.hoeffding_tree import HoeffdingTree
    from river.tree.nodes.branch import DTBranch
    from river.tree.nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
    from river.tree.nodes.leaf import HTLeaf
    from river.tree.split_criterion import GiniSplitCriterion, HellingerDistanceCriterion, InfoGainSplitCriterion
    from river.tree.splitter import GaussianSplitter, Splitter
    from river.tree.splitter.nominal_splitter_classif import NominalSplitterClassif
    from river.utils.norm import normalize_values_in_dict
    from river.tree.utils import do_naive_bayes_prediction, round_sig_fig
    from river.base.typing import ClfTarget
    from river import base
except ImportError:
    print("The 'river' package is required. Install with: pip install river")
    print("This package provides online machine learning components.")
    class metrics:
        class Accuracy:
            def update(self, *args): pass
            def get(self): return 0
        class CohenKappa(Accuracy): pass
        class MacroF1(Accuracy): pass
        class MicroF1(Accuracy): pass
        class MacroPrecision(Accuracy): pass
        class MacroRecall(Accuracy): pass
        class MicroPrecision(Accuracy): pass
        class MicroRecall(Accuracy): pass
        class WeightedF1(Accuracy): pass
        class WeightedPrecision(Accuracy): pass
        class WeightedRecall(Accuracy): pass
        class Recall(Accuracy): pass
        class F1(Accuracy): pass
        class Precision(Accuracy): pass


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        return super(NumpyEncoder, self).default(obj)



class MonotonicityConstraint:
    """
    Defines a monotonicity constraint on a specific feature.
    """
    def __init__(self, feature, monotonicity):
        """
        Initialize a monotonicity constraint.
        
        Parameters:
        -----------
        feature : int or str
            Feature index or name on which to apply the constraint
        monotonicity : int
            Direction of monotonicity: 1 for positive, -1 for negative
        """
        self.feature = feature
        self.monotonicity = monotonicity
    
    def __str__(self):
        direction = "positive" if self.monotonicity == 1 else "negative"
        return f"Feature {self.feature}, {direction} monotonicity"
    
    def __repr__(self):
        return self.__str__()



class HTLeafMono(Leaf, abc.ABC):
    """Base leaf class to be used in Hoeffding Trees with monotonicity constraints.

    Parameters
    ----------
    stats
        Target statistics (they differ in classification and regression tasks).
    depth
        The depth of the node
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attributes
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(**kwargs)
        self.stats = stats
        self.depth = depth

        self.splitter = splitter

        self.splitters = {}
        self._disabled_attrs = set()
        self._last_split_attempt_at = self.total_weight

    @property
    @abc.abstractmethod
    def total_weight(self) -> float:
        pass

    def is_active(self):
        return self.splitters is not None

    def activate(self):
        if not self.is_active():
            self.splitters = {}

    def deactivate(self):
        self.splitters = None

    @property
    def last_split_attempt_at(self) -> float:
        """The weight seen at last split evaluation.

        Returns
        -------
        Weight seen at last split evaluation.
        """
        return self._last_split_attempt_at

    @last_split_attempt_at.setter
    def last_split_attempt_at(self, weight):
        """Set the weight seen at last split evaluation.

        Parameters
        ----------
        weight
            Weight seen at last split evaluation.
        """
        self._last_split_attempt_at = weight

    @staticmethod
    @abc.abstractmethod
    def new_nominal_splitter():
        pass

    @abc.abstractmethod
    def update_stats(self, y, w):
        pass

    def _iter_features(self, x) -> typing.Iterable:
        """Determine how the input instance is looped through when updating the splitters.

        Parameters
        ----------
        x
            The input instance.
        """
        yield from x.items()

    def update_splitters(self, x, y, w, nominal_attributes):
        for att_id, att_val in self._iter_features(x):
            if att_id in self._disabled_attrs:
                continue

            try:
                splitter = self.splitters[att_id]
            except KeyError:
                if (
                    nominal_attributes is not None and att_id in nominal_attributes
                ) or not isinstance(att_val, numbers.Number):
                    splitter = self.new_nominal_splitter()
                else:
                    splitter = self.splitter.clone()

                self.splitters[att_id] = splitter
            splitter.update(att_val, y, w)

    def best_split_suggestions(self, criterion, tree) -> list[BranchFactory]:
        """Find possible split candidates.

        Parameters
        ----------
        criterion
            The splitting criterion to be used.
        tree
            Decision tree.

        Returns
        -------
        Split candidates.
        """
        leafs = tree._find_leaves()
        clean_leafs = [leaf for leaf in leafs if leaf is not self]

        best_suggestions = []
        pre_split_dist = self.stats
        if tree.merit_preprune:
            null_split = BranchFactory()
            best_suggestions.append(null_split)
        for att_id, splitter in self.splitters.items():
            best_suggestion = splitter.best_evaluated_split_suggestion(
                criterion, pre_split_dist, att_id, tree.binary_split, clean_leafs
            )
            best_suggestions.append(best_suggestion)

        return best_suggestions

    def disable_attribute(self, att_id):
        """Disable an attribute observer.

        Parameters
        ----------
        att_id
            Attribute index.

        """
        if att_id in self.splitters:
            del self.splitters[att_id]
            self._disabled_attrs.add(att_id)

    def learn_one(self, x, y, *, w=1.0, tree=None):
        """Update the node with the provided sample.

        Parameters
        ----------
        x
            Sample attributes for updating the node.
        y
            Target value.
        w
            Sample weight.
        tree
            Tree to update.

        Notes
        -----
        This base implementation defines the basic functioning of a learning node.
        All classes overriding this method should include a call to `super().learn_one`
        to guarantee the learning process happens consistently.
        """
        self.update_stats(y, w)
        if self.is_active():
            self.update_splitters(x, y, w, tree.nominal_attributes)

    @abc.abstractmethod
    def prediction(self, x, *, tree=None) -> dict:
        pass

    @abc.abstractmethod
    def calculate_promise(self) -> int:
        """Calculate node's promise."""
        pass

class LeafMajorityClassMono(HTLeafMono):
    """Leaf that always predicts the majority class with monotonicity constraint.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)

    @staticmethod
    def new_nominal_splitter():
        return NominalSplitterClassif()

    def update_stats(self, y, w):
        try:
            self.stats[y] += w
        except KeyError:
            self.stats[y] = w

    def prediction(self, x, *, tree=None):
        return normalize_values_in_dict(self.stats, inplace=False)

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
            Total weight seen.

        """
        return sum(self.stats.values()) if self.stats else 0

    def best_split_suggestions(self, criterion, tree) -> list[BranchFactory]:
        maj_class = max(self.stats.values())
        if maj_class and maj_class / self.total_weight > tree.max_share_to_split:
            return [BranchFactory()]

        return super().best_split_suggestions(criterion, tree)

    def calculate_promise(self):
        """Calculate how likely a node is going to be split.

        A node with a (close to) pure class distribution will less likely be split.

        Returns
        -------
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
        total_seen = sum(self.stats.values())
        if total_seen > 0:
            return total_seen - max(self.stats.values())
        else:
            return 0

    def observed_class_distribution_is_pure(self):
        """Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
            True if observed number of classes is less than 2, False otherwise.
        """
        count = 0
        for weight in self.stats.values():
            if weight != 0:
                count += 1
                if count == 2:
                    break
        return count < 2

    def __repr__(self):
        if not self.stats:
            return ""

        text = f"Class {max(self.stats, key=self.stats.get)}:"
        for label, proba in sorted(normalize_values_in_dict(self.stats, inplace=False).items()):
            text += f"\n\tP({label}) = {round_sig_fig(proba)}"

        return text

class DummyDist:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

class DummySplitter(GaussianSplitter):
    def __init__(self, n_splits: int = 10):
        super().__init__(n_splits)
        self._att_dist_per_class = {}
    
    def build_distributions(self, distributions, split_value, side, minpc, maxpc): #side 0 - lhs, 1 - rhs
        for class_idx, dist in distributions.items():
            self._att_dist_per_class[class_idx] = DummyDist(dist.mu, dist.sigma)

            if side == 0: #lt split value
                span_left = minpc[class_idx]
                span_right = split_value
            else: #gt split value
                span_left = split_value
                span_right = maxpc[class_idx]

            self._att_dist_per_class[class_idx].mu = (span_right - span_left) / 2
            self._att_dist_per_class[class_idx].sigma = (span_right - span_left) / 6
            if self._att_dist_per_class[class_idx].sigma < 0:
                self._att_dist_per_class[class_idx].sigma = 0

class CustomGaussianSplitter(GaussianSplitter):
    def __init__(self, n_splits: int = 10, monotonicity_constraints: list[MonotonicityConstraint] = [], rho: float = 1e-5, stdevmul: float = 2.0, max_correction_factor: int = 10000):
        super().__init__(n_splits=n_splits)
        self.monotonicity_constraints = monotonicity_constraints
        self.rho = rho
        self.stdevmul = stdevmul
        self.max_correction_factor = max_correction_factor
    

    
    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only, leafs):
        best_suggestion = BranchFactory()
        suggested_split_values = self._split_point_suggestions()
        
        for split_value in suggested_split_values:
            post_split_dist = self._class_dists_from_binary_split(split_value)
            
            base_merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
            
            correction = self._calculate_monotonicity_correction_factor(att_idx, split_value, pre_split_dist, post_split_dist, leafs)
            
            if correction > 0 and base_merit > 0:

                scaled_correction = min(base_merit, self.max_correction_factor * correction)
                
                merit = max(0, base_merit + scaled_correction)
            else:
                merit = base_merit
            
            
            if merit > best_suggestion.merit:
                best_suggestion = BranchFactory(merit, att_idx, split_value, post_split_dist)

        return best_suggestion
    
    def _calculate_monotonicity_correction_factor(self, att_idx, split_value, pre_split_dist, post_split_dist, leafs):
        simulatedBranchFactory = BranchFactory(-math.inf, att_idx, split_value, post_split_dist)
        simleafs = []
        if len(simulatedBranchFactory.children_stats) == 0:
            return 0
        
        self_clone = self.clone()
        self_clone._att_dist_per_class = self._att_dist_per_class
        
        cnt = 0
        for psd in simulatedBranchFactory.children_stats:
            ds = DummySplitter(self.n_splits)
            ds.build_distributions(self._att_dist_per_class, split_value, cnt, self._min_per_class, self._max_per_class)
            simleafs.append(LeafMajorityClassMono(psd, 1, ds))
            cnt += 1
            
        total_leaves = len(leafs) + len(simleafs)
        matrix = np.zeros((total_leaves, total_leaves))  # ordering leaves then simulated leaves in the array order
        denom = total_leaves * (total_leaves - 1)
        if denom == 0:
            return 0
        for i, leaf in enumerate(leafs + simleafs):
            for j, leaf2 in enumerate(leafs + simleafs):
                if i == j or i > j: # assymetric matrix
                    continue
                if self._is_monotonic(leaf, leaf2, att_idx):
                    matrix[i, j] = 1
        W = np.sum(matrix)
        I = W / denom
        Av = self._calculate_A(I)
        return self.rho * Av

    def _is_monotonic(self, leaf1, leaf2, att_idx):
        val = 1
        mcs = self.monotonicity_constraints
        for mc in mcs:
            if mc.feature == att_idx:  # do we have a constraint for this feature
                if mc.monotonicity == 1:
                    val = val and self._is_increasing(leaf1, leaf2)  # 1 when leaf1 nonmonotone wrt leaf2
                    if val == 0:
                        return 1
                else:
                    val = val and self._is_decreasing(leaf1, leaf2)  # 1 when leaf1 nonmonotone wrt leaf2
                    if val == 0:
                        return 1
        return 1 - val
    
    def _is_increasing(self, leaf1, leaf2):  # substituting minmax values with distribution + n*stdev
        if len(leaf1.splitter._att_dist_per_class) == 0 or len(leaf2.splitter._att_dist_per_class) == 0:  # fresh leaf has no distribution
            return 0
        l1p = leaf1.prediction(None)
        l2p = leaf2.prediction(None)
        max_l1p = max(l1p, key=lambda k: float(l1p[k]))
        max_l2p = max(l2p, key=lambda k: float(l2p[k]))
        
        l1distr = leaf1.splitter._att_dist_per_class[max_l1p]
        l2distr = leaf2.splitter._att_dist_per_class[max_l2p]
        
        return max_l1p > max_l2p and (l1distr.mu - l1distr.sigma * self.stdevmul) < (l2distr.mu + l2distr.sigma * self.stdevmul)
    
    def _is_decreasing(self, leaf1, leaf2):
        if len(leaf1.splitter._att_dist_per_class) == 0 or len(leaf2.splitter._att_dist_per_class) == 0:  # fresh leaf has no distribution
            return 0
        l1p = leaf1.prediction(None)
        l2p = leaf2.prediction(None)
        max_l1p = max(l1p, key=lambda k: float(l1p[k]))
        max_l2p = max(l2p, key=lambda k: float(l2p[k]))
        l1distr = leaf1.splitter._att_dist_per_class[max_l1p]
        l2distr = leaf2.splitter._att_dist_per_class[max_l2p]
        return max_l1p < max_l2p and (l1distr.mu + l1distr.sigma * self.stdevmul) > (l2distr.mu - l2distr.sigma * self.stdevmul)
    
    def _calculate_A(self, I):
        if I == 1:
            return 0
        else:
            return -1 / np.log2(I)

class HoeffdingTreeClassifierMono(HoeffdingTree, base.Classifier):
    """Hoeffding Tree classifier with monotonicity constraints.

    Parameters
    ----------
    monotonic_constrains
        List of MonotonicityConstraint objects to apply to the tree.
    rho
        The strength of the monotonicity penalty.
    stdevmul
        Standard deviation multiplier for constraint checking.
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_criterion
        Split criterion to use.
    delta
        Significance level to calculate the Hoeffding bound.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers.
    splitter
        The Splitter or Attribute Observer used to monitor class statistics.
    binary_split
        If True, only allow binary splits.
    min_branch_fraction
        Minimum percentage of samples required for branches from split candidates.
    max_share_to_split
        Only perform a split if the proportion of elements in the majority class is
        smaller than this parameter value.
    max_size
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.
    """

    _GINI_SPLIT = "gini"
    _INFO_GAIN_SPLIT = "info_gain"
    _HELLINGER_SPLIT = "hellinger"
    _VALID_SPLIT_CRITERIA = [_GINI_SPLIT, _INFO_GAIN_SPLIT, _HELLINGER_SPLIT]

    _MAJORITY_CLASS = "mc"
    _NAIVE_BAYES = "nb"
    _NAIVE_BAYES_ADAPTIVE = "nba"
    _VALID_LEAF_PREDICTION = [_MAJORITY_CLASS, _NAIVE_BAYES, _NAIVE_BAYES_ADAPTIVE]

    def __init__(
        self,
        monotonic_constrains: list[MonotonicityConstraint],
        rho: float = 1e-5,
        stdevmul: float = 2.0,
        grace_period: int = 200,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        max_correction_factor: int = 10000
    ):
        super().__init__(
            max_depth=max_depth,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.rho = rho
        self.stdevmul = stdevmul
        self.tree_indecies = []
        self.tree_leafs = []
        self.constrains = monotonic_constrains
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        if splitter is None:

            self.splitter = CustomGaussianSplitter(n_splits=10, monotonicity_constraints=self.constrains, rho=self.rho, stdevmul=self.stdevmul, max_correction_factor=max_correction_factor)
        else:
            if not splitter.is_target_class:
                raise ValueError("The chosen splitter cannot be used in classification tasks.")
            self.splitter = splitter  # type: ignore

        self.min_branch_fraction = min_branch_fraction
        self.max_share_to_split = max_share_to_split

        self.classes: set = set()

    @property
    def _mutable_attributes(self):
        return {"grace_period", "delta", "tau"}

    @HoeffdingTree.split_criterion.setter  # type: ignore
    def split_criterion(self, split_criterion):
        if split_criterion not in self._VALID_SPLIT_CRITERIA:
            print(
                "Invalid split_criterion option {}', will use default '{}'".format(
                    split_criterion, self._INFO_GAIN_SPLIT
                )
            )
            self._split_criterion = self._INFO_GAIN_SPLIT
        else:
            self._split_criterion = split_criterion

    @HoeffdingTree.leaf_prediction.setter  # type: ignore
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in self._VALID_LEAF_PREDICTION:
            print(
                "Invalid leaf_prediction option {}', will use default '{}'".format(
                    leaf_prediction, self._NAIVE_BAYES_ADAPTIVE
                )
            )
            self._leaf_prediction = self._NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return LeafMajorityClassMono(initial_stats, depth, self.splitter)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return LeafNaiveBayes(initial_stats, depth, self.splitter)
        else:  # Naives Bayes Adaptive (default)
            return LeafNaiveBayesAdaptive(initial_stats, depth, self.splitter)

    def _new_split_criterion(self):
        if self._split_criterion == self._GINI_SPLIT:
            split_criterion = GiniSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._INFO_GAIN_SPLIT:
            split_criterion = InfoGainSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._HELLINGER_SPLIT:
            split_criterion = HellingerDistanceCriterion(self.min_branch_fraction)
        else:
            split_criterion = InfoGainSplitCriterion(self.min_branch_fraction)

        return split_criterion

    def _attempt_to_split(self, leaf: HTLeaf | HTLeafMono, parent: DTBranch, parent_branch: int, **kwargs):
        """Attempt to split a leaf with monotonicity constraints.
        """
        if not leaf.observed_class_distribution_is_pure():  # type: ignore
            split_criterion = self._new_split_criterion()

            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort()
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self.delta,
                    leaf.total_weight,
                )
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (
                    best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                    or hoeffding_bound < self.tau
                ):
                    should_split = True
                if self.remove_poor_attrs:
                    poor_atts = set()
                    for suggestion in best_split_suggestions:
                        if (
                            suggestion.feature
                            and best_suggestion.merit - suggestion.merit > hoeffding_bound
                        ):
                            poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.feature is None:
                    leaf.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats  # type: ignore
                    )

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._root = new_split
                    else:
                        parent.children[parent_branch] = new_split

                self._enforce_size_limit()

    def learn_one(self, x, y, *, w=1.0):
        """Train the model on instance x and corresponding target y.
        """
        self.classes.add(y)

        self._train_weight_seen_by_model += w

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        p_node = None
        node = None
        if isinstance(self._root, DTBranch):
            path = iter(self._root.walk(x, until_leaf=False))
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
        else:
            node = self._root

        if isinstance(node, HTLeaf) or isinstance(node, HTLeafMono):
            node.learn_one(x, y, w=w, tree=self)
            if self._growth_allowed and node.is_active():
                if node.depth >= self.max_depth:  # Max depth reached
                    node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = node.total_weight
                    weight_diff = weight_seen - node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        p_branch = p_node.branch_no(x) if isinstance(p_node, DTBranch) else None
                        self._attempt_to_split(node, p_node, p_branch)
                        node.last_split_attempt_at = weight_seen
        else:
            while True:
                if node.max_branches() == -1 and node.feature in x:
                    leaf = self._new_leaf(parent=node)
                    node.add_child(x[node.feature], leaf)
                    self._n_active_leaves += 1
                    node = leaf
                else:
                    _, node = node.most_common_path()
                    if isinstance(node, DTBranch):
                        node = node.traverse(x, until_leaf=False)
                if isinstance(node, HTLeaf):
                    break
            node.learn_one(x, y, w=w, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in sorted(self.classes)}
        if self._root is not None:
            if isinstance(self._root, DTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root

            proba.update(leaf.prediction(x, tree=self))
        return proba

    @property
    def _multiclass(self):
        return True



class UCIDatasetLoader:
    """
    Loader for UCI datasets with monotonicity constraints for experimentation.
    
    This class handles loading UCI datasets from a directory structure,
    applying appropriate preprocessing, and extracting features and targets
    according to specified monotonicity constraints.
    """
    
    def __init__(self, data_dir: str, constraints_file: Optional[str] = None):
        """
        Initialize the UCI dataset loader.
        
        Parameters:
        -----------
        data_dir : str
            Path to the directory containing UCI datasets
        constraints_file : str, optional
            Path to a CSV file containing constraint definitions
        """
        self.data_dir = data_dir
        self.default_constraints = self._define_default_constraints()
        
        self.custom_constraints = {}
        if constraints_file and os.path.exists(constraints_file):
            self._load_constraints_file(constraints_file)
    
    def _load_constraints_file(self, constraints_file: str):
        """Load constraints from CSV file with support for nested constraints."""
        try:
            constraints_df = pd.read_csv(constraints_file)
            custom_constraints = {}
            
            for _, row in constraints_df.iterrows():
                dataset = row['Dataset'].strip()
                
                positive_features = self._parse_list_field(row['Positive Monotonic Features'])
                negative_features = self._parse_list_field(row['Negative Monotonic Features'])
                
                custom_constraints[dataset] = {
                    'target': row['Target'].strip(),
                    'positive_monotonic': positive_features,
                    'negative_monotonic': negative_features,
                    'notes': row['Notes'] if 'Notes' in row else ''
                }
            
            self.custom_constraints = custom_constraints
        except Exception as e:
            warnings.warn(f"Error loading constraints file: {e}")
    
    def _parse_list_field(self, list_str):
        """
        Parse a string representation of a list into an actual list.
        This updated version handles nested lists.
        
        Examples:
        ---------
        "['a1', 'a2', ['a1', 'a2']]" -> ['a1', 'a2', ['a1', 'a2']]
        """
        if pd.isna(list_str) or not list_str:
            return []
        
        if isinstance(list_str, str):
            try:
                import ast
                return ast.literal_eval(list_str)
            except (SyntaxError, ValueError):
                cleaned = list_str.strip('[]').replace('"', '').replace("'", "")
                if cleaned:
                    items = []
                    for item in cleaned.split(','):
                        item = item.strip()
                        if item.startswith('[') and item.endswith(']'):
                            nested_items = item[1:-1].split()
                            items.append([i.strip() for i in nested_items])
                        else:
                            items.append(item)
                    return items
        
        return []
    
    def _define_default_constraints(self) -> Dict:
        """Define default monotonicity constraints for known UCI datasets."""
        return {
            "monks_1": {
                "target": "class",
                "positive_monotonic": ["a1", "a2", "a3", ["a1", "a2", "a3"]],
                "negative_monotonic": ["a4", "a5", "a6", ["a4", "a5", "a6"]],
                "notes": "Binary classification with logical rules"
            },
            "monks_2": {
                "target": "class",
                "positive_monotonic": ["a1", "a2", "a3", ["a1", "a2", "a3"]],
                "negative_monotonic": ["a4", "a5", "a6", ["a4", "a5", "a6"]],
                "notes": "Binary classification with logical rules"
            },
            "monks_3": {
                "target": "class",
                "positive_monotonic": ["a1", "a2", "a3", ["a1", "a2", "a3"]],
                "negative_monotonic": ["a4", "a5", "a6", ["a4", "a5", "a6"]],
                "notes": "Binary classification with logical rules"
            },
            "mammographic_mass": {
                "target": "Severity",
                "positive_monotonic": ["Age", ["Age"]],
                "negative_monotonic": ["Density", ["Density"]],
                "notes": "Binary classification, 1=benign, 0=malignant"
            },
            "iris": {
                "target": "Species",
                "positive_monotonic": ["petal length", "petal width", ["petal length", "petal width"]],
                "negative_monotonic": ["sepal width", ["sepal width"]],
                "notes": "Multiclass classification, ordering: setosa < versicolor < virginica"
            },
            "hepatitis": {
                "target": "Class",
                "positive_monotonic": ["Age", ["Age"]],
                "negative_monotonic": ["Albumin", ["Albumin"]],
                "notes": "Binary classification, 1=died, 0=lived (reversed for monotonicity)"
            },
            "wine": {
                "target": "class",
                "positive_monotonic": ["Alcohol", "Flavanoids", ["Alcohol", "Flavanoids"]],
                "negative_monotonic": ["Ash", ["Ash"]],
                "notes": "Multiclass classification, ordered by quality"
            },
            "heart_disease": {
                "target": "num",
                "positive_monotonic": ["thalach", ["thalach"]],  # Max Heart Rate
                "negative_monotonic": ["age", "chol", ["age", "chol"]],  # Age, Cholesterol
                "notes": "Binary classification, 0=no disease, 1=disease"
            },
            "adult": {
                "target": "income",
                "positive_monotonic": ["education-num", "hours-per-week", ["education-num", "hours-per-week"]],
                "negative_monotonic": ["age", ["age"]],
                "notes": "Binary classification, '>50K'-1 vs '<=50K'-0" 
            },
            "student_performance": {
                "target": "G3",  # Final grade
                "positive_monotonic": ["studytime", ["studytime"]],
                "negative_monotonic": ["absences", ["absences"]],
                "notes": "Regression/ordinal, final grade"
            },
            "diabetes": {
                "target": "Outcome",
                "positive_monotonic": ["Glucose", "BMI", ["Glucose", "BMI"]],
                "negative_monotonic": ["Age", ["Age"]],
                "notes": "Binary classification, 1=diabetes, 0=no diabetes"
            },
            "spambase": {
                "target": "class",
                "positive_monotonic": ["word_freq_free", "word_freq_money", ["word_freq_free", "word_freq_money"]], 
                "negative_monotonic": ["word_freq_hp", "word_freq_george", ["word_freq_hp", "word_freq_george"]],
                "notes": "Binary classification, 1=spam, 0=not spam"
            },
            "thoracic_surgery": {
                "target": "Risk1Yr",
                "positive_monotonic": ["FEV1", ["FEV1"]],
                "negative_monotonic": ["Age", ["Age"]],
                "notes": "Binary classification, 1=dead, 0=alive"
            },
            "air_quality": {
                "target": "CO_GT",  # CO level (discretized)
                "positive_monotonic": ["NOx", ["NOx"]],
                "negative_monotonic": ["O3", ["O3"]],
                "notes": "Regression/ordinal target"
            },
            "automobile": {
                "target": "price",
                "positive_monotonic": ["engine-size", ["engine-size"]],
                "negative_monotonic": ["highway-mpg", ["highway-mpg"]],
                "notes": "Regression/ordinal target"
            },
            "auto_mpg": {
                "target": "mpg",
                "positive_monotonic": [],
                "negative_monotonic": ["weight", "horsepower", ["weight", "horsepower"]],
                "notes": "Regression target (higher is better, so negative monotonicity)"
            },
            "ionosphere": {
                "target": "class",
                "positive_monotonic": [],
                "negative_monotonic": [],
                "notes": "Binary classification, 1=good, 0=bad"
            },
            "london": {
                "target": "travel_mode",
                "positive_monotonic": ["car_ownership", "distance", "driving_license", "pt_interchanges", ["car_ownership", "distance", "driving_license", "pt_interchanges"]],
                "negative_monotonic": ["car_ownership", "distance", "driving_license", "pt_interchanges", ["car_ownership", "distance", "driving_license", "pt_interchanges"]],
                "notes": "Binary classification, 1=car 0=else"
            },
            "students":{
                "target": "Car",  # Final grade
                "positive_monotonic": ["Distance","Season","CarAvail",["Distance","Season","CarAvail"]],
                "negative_monotonic": ["Age","Grade","Gender", ["Age","Grade","Gender"]],
                "notes": "1 car 0 something else"
            },
            "flights":{
                "target": "Cancelled",
                "positive_monotonic": ["Distance","Origin_Latitude","Month",["Distance","Airline","FlightNum"]],
                "negative_monotonic": ["Distance","Origin_Latitude",["DayOfWeek","Month","Year"]],
                "notes": "Binary classification, 1=cancelled, 0=not cancelled"
            },
            "ohio":{
                "target": "label",
                "positive_monotonic": ["drivers","workers","guest",[]],
                "negative_monotonic": ["income","students",[]],
                "notes": "Binary classification, 1=sole driver, 0=not a sole driver"
            }
        }
    
    def _get_constraints(self, dataset_name: str) -> Dict:
        """
        Get constraints for a specific dataset, prioritizing custom constraints if available.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        Dict with target and monotonicity constraints
        """
        for key in self.custom_constraints:
            if dataset_name.lower() in key.lower():
                return self.custom_constraints[key]
        print(f"Running {dataset_name}")
        if "monk" in dataset_name.lower():
            if "_1" in dataset_name or "-1" in dataset_name:
                return self.default_constraints["monks_1"]
            elif "_2" in dataset_name or "-2" in dataset_name:
                return self.default_constraints["monks_2"]
            elif "_3" in dataset_name or "-3" in dataset_name:
                return self.default_constraints["monks_3"]
            else:
                print(f"Using monks_1 constraints for {dataset_name}")
                return self.default_constraints["monks_1"]
        
        for key in self.default_constraints:
            if dataset_name.lower() in key.lower() or key.lower() in dataset_name.lower():
                print(f"Found match: {key} for {dataset_name}")
                return self.default_constraints[key]
        
        warnings.warn(f"No constraints found for dataset: {dataset_name}")
        return {
            "target": None,
            "positive_monotonic": [],
            "negative_monotonic": [],
            "notes": "No constraints defined"
        }
    
    def load_datasets(self, max_datasets: Optional[int] = None, 
                     dataset_filter: Optional[List[str]] = None) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """
        Load UCI datasets with constraints.
        
        Parameters:
        -----------
        max_datasets : int, optional
            Maximum number of datasets to load
        dataset_filter : List[str], optional
            List of dataset names to load (if None, load all)
            
        Returns:
        --------
        Dict of {dataset_name: (DataFrame, constraints_dict)}
        """
        print(f"Loading datasets from {self.data_dir}")
        dataset_dirs = [d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))]
        print(f"Found datasets: {dataset_dirs}")
        if dataset_filter:
            dataset_dirs = [d for d in dataset_dirs 
                          if any(filter_name.lower() in d.lower() for filter_name in dataset_filter)]
        print(f"Filtered datasets: {dataset_dirs}")
        if max_datasets:
            dataset_dirs = dataset_dirs[:max_datasets]
        
        loaded_datasets = {}
        for dataset_dir in dataset_dirs:
            try:
                normalized_name = self._normalize_dataset_name(dataset_dir)
                df, constraints = self._load_specific_dataset(normalized_name)
                
                if df is not None:
                    loaded_datasets[normalized_name] = (df, constraints)
            except Exception as e:
                warnings.warn(f"Error loading dataset {dataset_dir}: {e}")
        
        return loaded_datasets
    
    def _normalize_dataset_name(self, dataset_name: str) -> str:
        """Normalize dataset name for consistent referencing."""
        name = dataset_name.replace('+', '_')
        name = name.replace(' ', '_')
        return name.lower()
    
    def _load_specific_dataset(self, dataset_name: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Load a specific dataset with specialized handling based on dataset name.
        
        Parameters:
        -----------
        dataset_name : str
            Normalized name of the dataset
            
        Returns:
        --------
        Tuple of (DataFrame, constraints_dict)
        """
        constraints = self._get_constraints(dataset_name)
        
        if "monk" in dataset_name:
            return self._load_monks_dataset(dataset_name), constraints
        elif "mammographic_mass" in dataset_name:
            return self._load_mammographic_mass(), constraints
        elif "iris" in dataset_name:
            return self._load_iris(), constraints
        elif "hepatitis" in dataset_name:
            return self._load_hepatitis(), constraints
        elif "wine" in dataset_name:
            return self._load_wine(), constraints
        elif "heart_disease" in dataset_name:
            return self._load_heart_disease(), constraints
        elif "adult" in dataset_name:
            return self._load_adult(), constraints
        elif "student_performance" in dataset_name:
            return self._load_student_performance(), constraints
        elif "diabetes" in dataset_name:
            return self._load_diabetes(), constraints
        elif "spambase" in dataset_name:
            return self._load_spambase(), constraints
        elif "thoracic_surgery" in dataset_name:
            return self._load_thoracic_surgery(), constraints
        elif "air_quality" in dataset_name:
            return self._load_air_quality(), constraints
        elif "automobile" in dataset_name:
            return self._load_automobile(), constraints
        elif "auto_mpg" in dataset_name:
            return self._load_auto_mpg(), constraints
        elif "ionosphere" in dataset_name:
            return self._load_ionosphere(), constraints
        elif "london" in dataset_name:
            return self._load_london(), constraints
        elif "students" in dataset_name:
            return self._load_students(), constraints
        elif "flights" in dataset_name:
            return self._load_flights(), constraints
        elif "ohio" in dataset_name:
            return self._load_ohio(), constraints
        else:
            print(f"Loading generic dataset: {dataset_name}")
            try:
                original_dir = dataset_name.replace('_', '+')
                dataset_path = os.path.join(self.data_dir, original_dir)
                
                csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
                data_files = glob.glob(os.path.join(dataset_path, "*.data"))
                
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                elif data_files:
                    df = pd.read_csv(data_files[0], header=None)
                else:
                    warnings.warn(f"No suitable data file found for {dataset_name}")
                    return None, constraints
                
                return df, constraints
            except Exception as e:
                warnings.warn(f"Error loading dataset {dataset_name}: {e}")
                return None, constraints
    
    def _extract_zip_if_needed(self, dataset_name: str) -> None:
        """Extract zip file if the dataset directory doesn't exist."""
        original_dir = dataset_name.replace('_', '+')
        dataset_path = os.path.join(self.data_dir, original_dir)
        
        if not os.path.exists(dataset_path):
            zip_path = os.path.join(self.data_dir, f"{original_dir}.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
    
    def _load_monks_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load Monks Problems dataset."""
        self._extract_zip_if_needed("monk_s_problems")
        
        monks_num = None
        for i in range(1, 4):
            if f"monk_{i}" in dataset_name:
                monks_num = i
                break
        
        if monks_num is None:
            monks_num = 1  # Default to Monk 1
        
        dataset_path = os.path.join(self.data_dir, "monk+s+problems")
        train_file = os.path.join(dataset_path, f"monks-{monks_num}.train")
        
        if not os.path.exists(train_file):
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file == f"monks-{monks_num}.train":
                        train_file = os.path.join(root, file)
                        break
        
        columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
        df = pd.read_csv(train_file, sep=' ', names=columns)
        
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
        
        return df
    
    def _load_mammographic_mass(self) -> pd.DataFrame:
        """Load Mammographic Mass dataset."""
        self._extract_zip_if_needed("mammographic_mass")
        
        dataset_path = os.path.join(self.data_dir, "mammographic+mass")
        data_file = os.path.join(dataset_path, "mammographic_masses.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for mammographic mass")
        
        columns = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
        df = pd.read_csv(data_file, na_values=['?'], names=columns)
        
        df = df.dropna()
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _load_iris(self) -> pd.DataFrame:
        """Load Iris dataset."""
        self._extract_zip_if_needed("iris")
        
        dataset_path = os.path.join(self.data_dir, "iris")
        data_file = os.path.join(dataset_path, "iris.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for iris")
        
        columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Species']
        df = pd.read_csv(data_file, names=columns)
        
        species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        df['Species'] = df['Species'].map(species_mapping)
        
        return df
    
    def _load_hepatitis(self) -> pd.DataFrame:
        """Load Hepatitis dataset."""
        self._extract_zip_if_needed("hepatitis")
        
        dataset_path = os.path.join(self.data_dir, "hepatitis")
        data_file = os.path.join(dataset_path, "hepatitis.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for hepatitis")
        
        columns = [
            'Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 
            'Anorexia', 'Liver Big', 'Liver Firm', 'Spleen Palpable', 'Spiders', 
            'Ascites', 'Varices', 'Bilirubin', 'Alk Phosphate', 'SGOT', 'Albumin', 
            'Protime', 'Histology'
        ]
        
        df = pd.read_csv(data_file, na_values=['?'], names=columns)
        
        df = df.dropna()
        
        for col in df.columns:
            if col != 'Class':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Class'] = df['Class'].map({2: 0, 1: 1})
        
        return df
    
    def _load_wine(self) -> pd.DataFrame:
        """Load Wine dataset."""
        self._extract_zip_if_needed("wine")
        
        dataset_path = os.path.join(self.data_dir, "wine")
        data_file = os.path.join(dataset_path, "wine.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for wine")
        
        columns = [
            'class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 
            'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
            'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 
            'Proline'
        ]
        
        df = pd.read_csv(data_file, names=columns)
        
        df['class'] = df['class'] - 1
        
        return df
    
    def _load_heart_disease(self) -> pd.DataFrame:
        """Load Heart Disease dataset."""
        self._extract_zip_if_needed("heart_disease")
        
        dataset_path = os.path.join(self.data_dir, "heart+disease")
        
        data_file = None
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if "processed.cleveland.data" in file:
                    data_file = os.path.join(root, file)
                    break
        
        if not data_file:
            data_files = glob.glob(os.path.join(dataset_path, "**/*.data"), recursive=True)
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for heart disease")
        
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
        ]
        
        df = pd.read_csv(data_file, na_values=['?'], names=columns)
        
        df = df.dropna()
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
        
        return df
    
    def _load_ionosphere(self) -> pd.DataFrame:
        """Load Ionosphere dataset."""
        self._extract_zip_if_needed("ionosphere")
        
        dataset_path = os.path.join(self.data_dir, "ionosphere")
        data_file = os.path.join(dataset_path, "ionosphere.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for ionosphere")
        
        columns = [f'feature_{i}' for i in range(1, 35)] + ['class']
        df = pd.read_csv(data_file, names=columns)
        
        df['class'] = df['class'].map({'g': 1, 'b': 0})
        
        print(df.head())
        
        return df
    
    def _load_london(self) -> pd.DataFrame:
        """Load London dataset."""
        self._extract_zip_if_needed("london")
        
        dataset_path = os.path.join(self.data_dir, "london")
        data_file = os.path.join(dataset_path, "london_data_extended_by_mg.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for london")
        
        df = pd.read_csv(data_file)




        df['travel_mode'] = df['travel_mode'].apply(lambda x: 1 if x == 4 else 0)
        
        return df
    
    def _load_adult(self) -> pd.DataFrame:
        """Load Adult dataset."""
        self._extract_zip_if_needed("adult")
        
        dataset_path = os.path.join(self.data_dir, "adult")
        data_file = os.path.join(dataset_path, "adult.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for adult")
        
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]
        
        df = pd.read_csv(data_file, names=columns, sep=', ', engine='python')
        
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
        
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                           'relationship', 'race', 'sex', 'native-country']
        
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def _load_student_performance(self) -> pd.DataFrame:
        """Load Student Performance dataset."""
        self._extract_zip_if_needed("student_performance")
        
        dataset_path = os.path.join(self.data_dir, "student+performance")
        
        data_file = os.path.join(dataset_path, "student-mat.csv")
        
        if not os.path.exists(data_file):
            data_file = os.path.join(dataset_path, "student-por.csv")
            
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for student performance")
        
        df = pd.read_csv(data_file, sep=';')
        
        categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                           'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                           'nursery', 'higher', 'internet', 'romantic']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def _load_diabetes(self) -> pd.DataFrame:
        """Load Pima Indians Diabetes dataset."""
        data_file = os.path.join(self.data_dir, "diabetes.csv")
        
        if not os.path.exists(data_file):
            diabetes_dirs = glob.glob(os.path.join(self.data_dir, "*diabetes*"))
            if diabetes_dirs:
                for diabetes_dir in diabetes_dirs:
                    csv_files = glob.glob(os.path.join(diabetes_dir, "*.csv"))
                    if csv_files:
                        data_file = csv_files[0]
                        break
            
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Cannot find diabetes.csv file")
        
        df = pd.read_csv(data_file)
        
        return df
    
    def _load_spambase(self) -> pd.DataFrame:
        """Load Spambase dataset."""
        self._extract_zip_if_needed("spambase")
        
        dataset_path = os.path.join(self.data_dir, "spambase")
        data_file = os.path.join(dataset_path, "spambase.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for spambase")
        
        word_freq_cols = [f"word_freq_{w}" for w in [
            'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order',
            'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business',
            'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 'george',
            'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999', 'parts',
            'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table',
            'conference'
        ]]
        
        char_freq_cols = [f"char_freq_{c}" for c in [';', '(', '[', '!', '$', '#']]
        capital_run_cols = ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']
        
        columns = word_freq_cols + char_freq_cols + capital_run_cols + ['class']
        
        df = pd.read_csv(data_file, names=columns)
        
        return df
    
    def _load_thoracic_surgery(self) -> pd.DataFrame:
        """Load Thoracic Surgery dataset."""
        self._extract_zip_if_needed("thoracic_surgery_data")
        
        dataset_path = os.path.join(self.data_dir, "thoracic+surgery+data")
        data_file = os.path.join(dataset_path, "ThoraricSurgery.arff")
        
        if not os.path.exists(data_file):
            for ext in ['.arff', '.data', '.csv']:
                files = glob.glob(os.path.join(dataset_path, f"*{ext}"))
                if files:
                    data_file = files[0]
                    break
            else:
                raise FileNotFoundError(f"Cannot find data file for thoracic surgery")
        
        if data_file.endswith('.arff'):
            data_lines = []
            columns = []
            with open(data_file, 'r') as f:
                data_section = False
                for line in f:
                    line = line.strip()
                    if line.lower().startswith('@attribute'):
                        parts = line.split()
                        if len(parts) > 1:
                            attr_name = parts[1].strip("'")
                            columns.append(attr_name)
                    elif line.lower().startswith('@data'):
                        data_section = True
                    elif data_section and line and not line.startswith('%'):
                        data_lines.append(line)
            
            df = pd.DataFrame([line.split(',') for line in data_lines], columns=columns)
            print(df.head())
            
            binary_cols = [ 'PRE7', 'PRE8', 'PRE9', 'PRE10', 
                          'PRE11',  'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32']
            for col in binary_cols:
                if col in df.columns:
                    df[col] = df[col].map({'T': 1, 'F': 0})
            
            multivalue_cols = ['DGN', 'PRE6','PRE14']
            df["DGN"] = df["DGN"].map({'DGN1': 0, 'DGN2': 1, 'DGN3': 2, 'DGN4': 3, 'DGN5': 4, 'DGN6': 5, 'DGN8': 6})
            df["PRE6"] = df["PRE6"].map({'PRZ2': 0, 'PRZ1': 1, 'PRZ0': 2})
            df["PRE14"] = df["PRE14"].map({'OC11': 0, 'OC14': 1, 'OC12': 2, 'OC13': 3})
            
            numeric_cols = ['AGE','PRE4', 'PRE5']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'Risk1Yr' in df.columns:
                df['Risk1Yr'] = df['Risk1Yr'].map({'T': 1, 'F': 0})
            elif 'Class' in df.columns:
                df['Risk1Yr'] = df['Class'].map({'T': 1, 'F': 0})
                df = df.drop('Class', axis=1)
        else:
            df = pd.read_csv(data_file)
        
        return df
    
    def _load_air_quality(self) -> pd.DataFrame:
        """Load Air Quality dataset."""
        self._extract_zip_if_needed("air_quality")
        
        dataset_path = os.path.join(self.data_dir, "air+quality")
        data_file = os.path.join(dataset_path, "AirQualityUCI.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for air quality")
        
        df = pd.read_csv(data_file, sep=';', decimal=',')
        print(f"Loaded air quality data from {data_file}")
        print(df.head())
        
        if 'Date' in df.columns:
            df = df.drop('Date', axis=1)
        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)
        
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].replace(-200, np.nan)
        
        if 'CO(GT)' in df.columns:
            df = df.dropna(subset=['CO(GT)'])
        
        rename_dict = {
            'CO(GT)': 'CO_GT',
            'PT08.S1(CO)': 'PT08_S1_CO',
            'NMHC(GT)': 'NMHC_GT',
            'C6H6(GT)': 'C6H6_GT',
            'PT08.S2(NMHC)': 'PT08_S2_NMHC',
            'NOx(GT)': 'NOx',
            'PT08.S3(NOx)': 'PT08_S3_NOx',
            'NO2(GT)': 'NO2',
            'PT08.S4(NO2)': 'PT08_S4_NO2',
            'PT08.S5(O3)': 'PT08_S5_O3',
            'T': 'Temperature',
            'RH': 'RelativeHumidity',
            'AH': 'AbsoluteHumidity'
        }
        
        df = df.rename(columns={c: rename_dict.get(c, c) for c in df.columns})

        unnamed_cols = [col for col in df.columns if 'Unnamed:' in col]
        if unnamed_cols:
            df = df.drop(unnamed_cols, axis=1)
        

        if 'CO_GT' in df.columns:
            df['CO_GT'] = df['CO_GT'].apply(lambda x: 1 if x > 2 else 0)

        return df
    
    def _load_automobile(self) -> pd.DataFrame:
        """Load Automobile dataset."""
        self._extract_zip_if_needed("automobile")
        
        dataset_path = os.path.join(self.data_dir, "automobile")
        data_file = os.path.join(dataset_path, "imports-85.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for automobile")
        
        columns = [
            'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 
            'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 
            'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 
            'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 
            'peak-rpm', 'city-mpg', 'highway-mpg', 'price'
        ]
        
        df = pd.read_csv(data_file, na_values=['?'], names=columns)
        
        numeric_cols = ['symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 
                       'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
                       'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 
                       'highway-mpg', 'price']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['price'])  # Ensure target variable is present
        
        categorical_cols = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
                           'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders',
                           'fuel-system']
        
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def _load_auto_mpg(self) -> pd.DataFrame:
        """Load Auto MPG dataset."""
        self._extract_zip_if_needed("auto_mpg")
        
        dataset_path = os.path.join(self.data_dir, "auto+mpg")
        data_file = os.path.join(dataset_path, "auto-mpg.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for auto mpg")
        
        columns = [
            'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration', 'model year', 'origin', 'car name'
        ]
        
        df = pd.read_fwf(data_file, names=columns, na_values=['?'])
        
        numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                       'acceleration', 'model year', 'origin']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if 'car name' in df.columns:
            df = df.drop('car name', axis=1)
        
        return df
    
    def _load_students(self) -> pd.DataFrame:
        """Load Students dataset."""
        self._extract_zip_if_needed("students")
        
        dataset_path = os.path.join(self.data_dir, "students")
        data_file = os.path.join(dataset_path, "students_binarized.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for students")
        
        df = pd.read_csv(data_file, sep=',')
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        df = df.dropna()
        
        return df
    
        
    def _load_ohio(self) -> pd.DataFrame:
        """Load Students dataset."""
        self._extract_zip_if_needed("ohio")
        
        dataset_path = os.path.join(self.data_dir, "ohio")
        data_file = os.path.join(dataset_path, "AllOhioDataSorted_cleaned.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for students")
            

        
        df = pd.read_csv(data_file, sep=',')
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        df = df.dropna()

        if 'ML_region' in df.columns:
            df = df.drop('ML_region', axis=1)
        
        return df


    
    def _load_flights(self) -> pd.DataFrame:
        self._extract_zip_if_needed("flights")
        dataset_path = os.path.join(self.data_dir, "flights")
        data_file = os.path.join(dataset_path, "flights_small.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for students")
        
        df = pd.read_csv(data_file, sep=',')
        print(f"Loaded flights data from {data_file}")

        df['Cancelled'] = df['Cancelled'].apply(lambda x: 1 if x else 0)
        if 'FlightDate' in df.columns:
            df = df.drop('FlightDate', axis=1)
        if 'Diverted' in df.columns:
            df = df.drop('Diverted', axis=1)

        categorical_cols = ['Airline', 'Origin', 'Dest', 'Marketing_Airline_Network', 
                           'Operated_or_Branded_Code_Share_Partners', 'IATA_Code_Marketing_Airline', 'Operating_Airline', 'IATA_Code_Operating_Airline',
                           'OriginCityName', 'OriginState', 'OriginStateName', 'DestCityName',
                           'DestState', 'DestStateName', 'DepTimeBlk', 'ArrTimeBlk',]
        
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def prepare_dataset_for_experiment(self, df: pd.DataFrame, 
                                      constraints: Dict,
                                      normalize: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare a dataset for experimentation by extracting features, target,
        and formatting the constraints for the Hoeffding Tree.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        constraints : Dict
            Dictionary with target and monotonicity constraints
        normalize : bool
            Whether to normalize numeric features
            
        Returns:
        --------
        Tuple of (prepared_df, formatted_constraints_dict)
        """
        print(f"Preparing dataset with constraints")
        print(df.head())

        if df is None or constraints is None:
            return None, {}
        
        print(f"Constraints: {constraints}")
        target_col = constraints.get('target')
        print(f"Target column: {target_col}")
        if not target_col or target_col not in df.columns:
            for col in df.columns:
                print(f"Checking column: {col}, target_col: {target_col}")
                if col.lower() == target_col.lower():
                    target_col = col
                    break
            else:
                warnings.warn(f"Target column '{target_col}' not found in DataFrame")
                return None, {}
        
        pos_monotonic = constraints.get('positive_monotonic', [])
        neg_monotonic = constraints.get('negative_monotonic', [])
        
        valid_pos_monotonic = []
        valid_neg_monotonic = []
        
        for feature in pos_monotonic:
            if isinstance(feature, list):
                group = []
                for f in feature:
                    if f in df.columns:
                        group.append(f)
                    else:
                        for col in df.columns:
                            if col.lower() == f.lower():
                                group.append(col)
                                break
                if group:  # Only add if at least one feature in group is valid
                    valid_pos_monotonic.append(group)
            elif feature in df.columns:
                valid_pos_monotonic.append(feature)
            else:
                for col in df.columns:
                    if col.lower() == feature.lower():
                        valid_pos_monotonic.append(col)
                        break
                else:
                    warnings.warn(f"Positive monotonic feature '{feature}' not found in DataFrame")
        
        for feature in neg_monotonic:
            if isinstance(feature, list):
                group = []
                for f in feature:
                    if f in df.columns:
                        group.append(f)
                    else:
                        for col in df.columns:
                            if col.lower() == f.lower():
                                group.append(col)
                                break
                if group:  # Only add if at least one feature in group is valid
                    valid_neg_monotonic.append(group)
            elif feature in df.columns:
                valid_neg_monotonic.append(feature)
            else:
                for col in df.columns:
                    if col.lower() == feature.lower():
                        valid_neg_monotonic.append(col)
                        break
                else:
                    warnings.warn(f"Negative monotonic feature '{feature}' not found in DataFrame")
        
        prepared_df = df.copy()

        print(f"Prepared DataFrame with {len(prepared_df)} rows and {len(prepared_df.columns)} columns")
        print(df.head())
        prepared_df = prepared_df.dropna()

        print(f"Prepared DataFrame with {len(prepared_df)} rows and {len(prepared_df.columns)} columns")
        
        if normalize:
            print("Normalizing numeric features")
            scaler = MinMaxScaler()
            numeric_cols = prepared_df.select_dtypes(include=['float', 'int']).columns
            
            if target_col in numeric_cols:
                numeric_cols = [col for col in numeric_cols if col != target_col]
            
            if len(numeric_cols) > 0:
                prepared_df[numeric_cols] = scaler.fit_transform(prepared_df[numeric_cols])
        
        formatted_constraints = {
            'target': target_col,
            'positive_monotonic': valid_pos_monotonic,
            'negative_monotonic': valid_neg_monotonic
        }
        
        return prepared_df, formatted_constraints
    
    def convert_to_experiment_format(self, df: pd.DataFrame, 
                                    constraints: Dict) -> Tuple[List[Dict], List, List[MonotonicityConstraint]]:
        """
        Convert DataFrame and constraints to a format compatible with the Hoeffding Tree.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Prepared DataFrame
        constraints : Dict
            Dictionary with target and monotonicity constraints
            
        Returns:
        --------
        Tuple of (data_stream, classes, monotonicity_constraints)
        """
        if df is None or constraints is None:
            return [], [], []
        
        target_col = constraints.get('target')
        if not target_col or target_col not in df.columns:
            warnings.warn(f"Target column '{target_col}' not found in DataFrame")
            return [], [], []
        
        print(df.head())
        
        features = df.drop(columns=[target_col]).reset_index(drop=True)
        feature_names = features.columns.tolist()
        target = df[target_col].reset_index(drop=True)
        
        data_stream = []
        for i, row in features.iterrows():
            X = {j: row[feature_names[j]] for j in range(len(feature_names))}
            y = target.iloc[i]
            data_stream.append((X, y))
        
        classes = sorted(target.unique().tolist())
        
        monotonicity_constraints = []
        
        for feature in constraints.get('positive_monotonic', []):
            if isinstance(feature, list):
                for f in feature:
                    if f in features.columns:
                        feature_idx = features.columns.get_loc(f)
                        monotonicity_constraints.append(MonotonicityConstraint(feature_idx, 1))
            else:
                if feature in features.columns:
                    feature_idx = features.columns.get_loc(feature)
                    monotonicity_constraints.append(MonotonicityConstraint(feature_idx, 1))
        
        for feature in constraints.get('negative_monotonic', []):
            if isinstance(feature, list):
                for f in feature:
                    if f in features.columns:
                        feature_idx = features.columns.get_loc(f)
                        monotonicity_constraints.append(MonotonicityConstraint(feature_idx, -1))
            else:
                if feature in features.columns:
                    feature_idx = features.columns.get_loc(feature)
                    monotonicity_constraints.append(MonotonicityConstraint(feature_idx, -1))
        
        return data_stream, classes, monotonicity_constraints
    
    def create_constraints_csv(self, output_path: str, custom_constraints: Optional[Dict] = None):
        """
        Create a constraints CSV file with default or custom constraints.
        Updated to handle nested constraints.
        
        Parameters:
        -----------
        output_path : str
            Path to save the constraints CSV file
        custom_constraints : Dict, optional
            Custom constraints to override defaults
        """
        constraints = custom_constraints if custom_constraints else self.default_constraints
        
        rows = []
        for dataset, constraint in constraints.items():
            rows.append({
                'Dataset': dataset,
                'Target': constraint.get('target', ''),
                'Positive Monotonic Features': str(constraint.get('positive_monotonic', [])),
                'Negative Monotonic Features': str(constraint.get('negative_monotonic', [])),
                'Notes': constraint.get('notes', '')
            })
        
        constraints_df = pd.DataFrame(rows)
        print(output_path)
        constraints_df.to_csv(output_path, index=False)
        
        print(f"Constraints CSV saved to: {output_path}")



def expand_constraints(constraints):
    """
    Expands constraint combinations from nested lists.
    
    Parameters:
    -----------
    constraints : dict
        Dictionary with 'target', 'positive_monotonic', and 'negative_monotonic' keys
        where the monotonic lists can contain both single features and lists of features
    
    Returns:
    --------
    list : List of constraint dictionaries, one for each combination
    """
    expanded = []
    target = constraints.get('target')
    pos_monotonic = constraints.get('positive_monotonic', [])
    neg_monotonic = constraints.get('negative_monotonic', [])
    
    pos_singles = [item for item in pos_monotonic if not isinstance(item, list)]
    pos_groups = [item for item in pos_monotonic if isinstance(item, list)]
    
    neg_singles = [item for item in neg_monotonic if not isinstance(item, list)]
    neg_groups = [item for item in neg_monotonic if isinstance(item, list)]
    
    for feature in pos_singles:
        expanded.append({
            'target': target,
            'positive_monotonic': [feature],
            'negative_monotonic': [],
            'experiment_name': f"{feature}(+)"
        })
    
    for feature in neg_singles:
        expanded.append({
            'target': target,
            'positive_monotonic': [],
            'negative_monotonic': [feature],
            'experiment_name': f"{feature}(-)"
        })
    
    for group in pos_groups:
        expanded.append({
            'target': target,
            'positive_monotonic': group,
            'negative_monotonic': [],
            'experiment_name': f"{'+'.join(group)}(+)"
        })
    
    for group in neg_groups:
        expanded.append({
            'target': target,
            'positive_monotonic': [],
            'negative_monotonic': group,
            'experiment_name': f"{'+'.join(group)}(-)"
        })
    
    if pos_monotonic and neg_monotonic:
        expanded.append({
            'target': target,
            'positive_monotonic': pos_monotonic,
            'negative_monotonic': neg_monotonic,
            'experiment_name': "All Constraints"
        })
    
    if pos_monotonic or neg_monotonic:
        for rho in [1e-6, 1e-5, 1e-4]:
            expanded.append({
                'target': target,
                'positive_monotonic': pos_monotonic,
                'negative_monotonic': neg_monotonic,
                'experiment_name': f"All Constraints (rho={rho})",
                'rho': rho
            })
    
    expanded.append({
        'target': target,
        'positive_monotonic': [],
        'negative_monotonic': [],
        'experiment_name': "No Constraints"
    })
    
    return expanded

def run_single_experiment(dataset_name, df, constraints, experiment_name, **kwargs):
    """
    Run an experiment with a single constraint configuration using Hoeffding Tree.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    df : pd.DataFrame
        Prepared DataFrame
    constraints : dict
        Dictionary with 'target', 'positive_monotonic', and 'negative_monotonic' keys
    experiment_name : str
        Name for this experiment configuration
    **kwargs :
        Additional parameters for the experiment
    
    Returns:
    --------
    dict : Results from this experiment
    """
    output_dir = kwargs.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    loader = UCIDatasetLoader("")  # Dummy loader for utility functions
    do_normalize = kwargs.get('normalize', True)
    prepared_df, formatted_constraints = loader.prepare_dataset_for_experiment(df, constraints, do_normalize)
    
    if prepared_df is None:
        print(f"Could not prepare dataset {dataset_name} for experiment")
        return None
    
    data_stream, classes, monotonicity_constraints = loader.convert_to_experiment_format(
        prepared_df, formatted_constraints)
    
    rho = kwargs.get('rho', 1e-5)
    stdevmul = kwargs.get('stdevmul', 2.0)
    grace_period = kwargs.get('grace_period', 200)
    max_depth = kwargs.get('max_depth', None)
    max_correction_factor = kwargs.get('max_correction_factor', 1.0)
    
    ht_with_constraints = HoeffdingTreeClassifierMono(
        monotonic_constrains=monotonicity_constraints,
        rho=rho,
        stdevmul=stdevmul,
        grace_period=grace_period,
        max_depth=max_depth,
        leaf_prediction="mc",
        split_criterion="info_gain",
        max_correction_factor=max_correction_factor
    )
    
    ht_without_constraints = HoeffdingTreeClassifierMono(
        monotonic_constrains=[],  # Empty list = no constraints
        rho=0,  # Set rho to 0 to disable constraint effects
        grace_period=grace_period,
        max_depth=max_depth,
        leaf_prediction="mc",
        split_criterion="info_gain",
        max_correction_factor=max_correction_factor
    )
    
    metrics_dict = {
        'accuracy': metrics.Accuracy(),
        'cohenkappa': metrics.CohenKappa(),
        'f1': metrics.F1(),
        'precision': metrics.Precision(),
        'recall': metrics.Recall(),
        'macrof1': metrics.MacroF1(),
        'macroprecision': metrics.MacroPrecision(),
        'macrorecall': metrics.MacroRecall()
    }
    
    results = {
        'dataset': dataset_name,
        'experiment_name': experiment_name,
        'rho': rho,
        'stdevmul': stdevmul,
        'grace_period': grace_period,
        'max_depth': max_depth,
        'iterations': [],
        'models': {
            'with_constraints': {
                'name': f'{experiment_name}',
                'online_predictions': [],
                'online_truths': []
            },
            'without_constraints': {
                'name': 'No Constraints',
                'online_predictions': [],
                'online_truths': []
            }
        }
    }
    
    for model_key in results['models']:
        for metric_name in metrics_dict:
            results['models'][model_key][f'online_{metric_name}'] = []
            results['models'][model_key][f'online_{metric_name}_obj'] = metrics_dict[metric_name].__class__()
    
    for i, (X, y) in enumerate(tqdm(data_stream, desc=f"Running {experiment_name}")):
        results['iterations'].append(i)
        
        pred_with = ht_with_constraints.predict_one(X)
        results['models']['with_constraints']['online_predictions'].append(pred_with)
        results['models']['with_constraints']['online_truths'].append(y)
        
        pred_without = ht_without_constraints.predict_one(X)
        results['models']['without_constraints']['online_predictions'].append(pred_without)
        results['models']['without_constraints']['online_truths'].append(y)
        
        for model_key, pred in [('with_constraints', pred_with), ('without_constraints', pred_without)]:
            for metric_name in metrics_dict:
                metric_key = f'online_{metric_name}_obj'
                results['models'][model_key][metric_key].update(y, pred)
                results['models'][model_key][f'online_{metric_name}'].append(
                    results['models'][model_key][metric_key].get()
                )
        
        ht_with_constraints.learn_one(X, y)
        ht_without_constraints.learn_one(X, y)
    
    for model_key, model in [('with_constraints', ht_with_constraints), 
                             ('without_constraints', ht_without_constraints)]:
        results['models'][model_key]['active_leaves'] = model._n_active_leaves
        results['models'][model_key]['inactive_leaves'] = model._n_inactive_leaves
        results['models'][model_key]['total_weight'] = model._train_weight_seen_by_model
    
    for model_key in results['models']:
        for metric_name in metrics_dict:
            del results['models'][model_key][f'online_{metric_name}_obj']
    
    file_name = f"{dataset_name}_{experiment_name.replace(' ', '_')}_results.json"
    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)
    
    visualize_experiment_results(results, os.path.join(output_dir, "plots"))
    
    return results

def run_multiple_experiments(dataset_name, df, constraints, **kwargs):
    """
    Run multiple experiments with different constraint combinations for Hoeffding Tree.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    df : pd.DataFrame
        Prepared DataFrame
    constraints : dict
        Dictionary with 'target', 'positive_monotonic', and 'negative_monotonic' keys
    **kwargs : 
        Additional parameters for the experiment
    
    Returns:
    --------
    dict : Combined results from all experiments
    """
    expanded_configs = []
    
    expanded_configs.append({
        'target': constraints['target'],
        'positive_monotonic': [],
        'negative_monotonic': [],
        'experiment_name': "No Constraints"
    })
    
    if constraints['positive_monotonic'] or constraints['negative_monotonic']:
        expanded_configs.append({
            'target': constraints['target'],
            'positive_monotonic': constraints['positive_monotonic'],
            'negative_monotonic': constraints['negative_monotonic'],
            'experiment_name': "All Constraints"
        })
    
    if constraints['positive_monotonic']:
        expanded_configs.append({
            'target': constraints['target'],
            'positive_monotonic': constraints['positive_monotonic'],
            'negative_monotonic': [],
            'experiment_name': "Positive Constraints Only"
        })
    
    if constraints['negative_monotonic']:
        expanded_configs.append({
            'target': constraints['target'],
            'positive_monotonic': [],
            'negative_monotonic': constraints['negative_monotonic'],
            'experiment_name': "Negative Constraints Only"
        })
    
    for feature in constraints['positive_monotonic']:
        if not isinstance(feature, list):
            expanded_configs.append({
                'target': constraints['target'],
                'positive_monotonic': [feature],
                'negative_monotonic': [],
                'experiment_name': f"Positive Constraint on {feature}"
            })
    
    for feature in constraints['negative_monotonic']:
        if not isinstance(feature, list):
            expanded_configs.append({
                'target': constraints['target'],
                'positive_monotonic': [],
                'negative_monotonic': [feature],
                'experiment_name': f"Negative Constraint on {feature}"
            })
    
    if constraints['positive_monotonic'] or constraints['negative_monotonic']:
        for rho in [1e-6, 1e-5, 1e-4]:
            expanded_configs.append({
                'target': constraints['target'],
                'positive_monotonic': constraints['positive_monotonic'],
                'negative_monotonic': constraints['negative_monotonic'],
                'experiment_name': f"All Constraints (rho={rho})",
                'rho': rho
            })
    
    combined_results = {
        'dataset': dataset_name,
        'iterations': [],
        'models': {}
    }
    
    output_dir = kwargs.get('output_dir', 'results')
    
    for config in expanded_configs:
        print(f"\nRunning experiment on dataset '{dataset_name}' with: {config['experiment_name']}")
        
        config_kwargs = kwargs.copy()
        for key in ['rho', 'stdevmul', 'grace_period', 'max_depth', 'max_correction_factor']:
            if key in config:
                config_kwargs[key] = config[key]
        
        result = run_single_experiment(
            dataset_name=dataset_name,
            df=df,
            constraints=config,
            experiment_name=config['experiment_name'],
            **config_kwargs
        )
        
        if result:
            if not combined_results['iterations'] and 'iterations' in result:
                combined_results['iterations'] = result['iterations']
            
            for model_key, model_data in result['models'].items():
                constraint_key = f"{model_key}_{config['experiment_name'].replace(' ', '_')}"
                combined_results['models'][constraint_key] = model_data
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    with open(os.path.join(output_dir, f"{dataset_name}_combined_results.json"), 'w') as f:
        json.dump(combined_results, f, cls=NumpyEncoder)
    
    visualize_combined_results(combined_results, os.path.join(output_dir, "plots"))
    
    return combined_results

def visualize_experiment_results(results, plots_dir):
    """
    Visualize results from a single experiment with both models.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_single_experiment
    plots_dir : str
        Directory to save plots
    """
    dataset_name = results['dataset']
    experiment_name = results['experiment_name']
    
    metrics_to_plot = ['accuracy', 'f1', 'macrof1', 'precision', 'recall']
    
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        for model_key, model_data in results['models'].items():
            metric_key = f'online_{metric_name}'
            
            if metric_key in model_data and len(model_data[metric_key]) > 0:
                subsample_rate = max(1, len(results['iterations']) // 100)
                x_values = results['iterations'][::subsample_rate]
                y_values = model_data[metric_key][::subsample_rate]
                
                linestyle = '-' if 'with_constraints' in model_key else '--'
                color = 'blue' if 'with_constraints' in model_key else 'red'
                
                plt.plot(
                    x_values, 
                    y_values, 
                    label=model_data['name'],
                    linestyle=linestyle,
                    color=color,
                    linewidth=2,
                    alpha=0.8
                )
        
        plt.title(f'Online {metric_name.capitalize()} - {dataset_name} - {experiment_name}')
        plt.xlabel('Iterations (samples seen)')
        plt.ylabel(f'{metric_name.capitalize()}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{experiment_name.replace(' ', '_')}_{metric_name}.png"))
        plt.close()
    
    plt.figure(figsize=(10, 6))
    
    metrics_for_comparison = ['accuracy', 'f1', 'macrof1', 'precision', 'recall']
    model_keys = list(results['models'].keys())
    model_names = [results['models'][k]['name'] for k in model_keys]
    
    bar_width = 0.35
    x = np.arange(len(metrics_for_comparison))
    
    for i, model_key in enumerate(model_keys):
        values = []
        
        for metric_name in metrics_for_comparison:
            metric_key = f'online_{metric_name}'
            if metric_key in results['models'][model_key] and len(results['models'][model_key][metric_key]) > 0:
                values.append(results['models'][model_key][metric_key][-1])
            else:
                values.append(0)
        
        offset = i * bar_width - (len(model_keys) * bar_width / 2) + bar_width/2
        bars = plt.bar(x + offset, values, bar_width, label=model_names[i], 
                      alpha=0.7, color='blue' if 'with_constraints' in model_key else 'red')
        
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{height:.3f}',
                ha='center', 
                va='bottom',
                fontsize=8
            )
    
    plt.xticks(x, metrics_for_comparison)
    plt.title(f'Final Metrics Comparison - {dataset_name} - {experiment_name}')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{experiment_name.replace(' ', '_')}_final_metrics.png"))
    plt.close()

def visualize_combined_results(results, plots_dir):
    """
    Visualize combined results from multiple constraint experiments.
    
    Parameters:
    -----------
    results : dict
        Combined results dictionary from run_multiple_experiments
    plots_dir : str
        Directory to save plots
    """
    dataset_name = results['dataset']
    
    metrics_to_plot = ['accuracy', 'f1', 'macrof1']
    
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.tab10.colors
        color_idx = 0
        
        constrained_models = {k: v for k, v in results['models'].items() 
                            if 'with_constraints' in k and 'No_Constraints' not in k}
        
        baseline_models = {k: v for k, v in results['models'].items() 
                         if 'without_constraints' in k or 'No_Constraints' in k}
        
        for model_key, model_data in baseline_models.items():
            metric_key = f'online_{metric_name}'
            
            if metric_key in model_data and len(model_data[metric_key]) > 0:
                subsample_rate = max(1, len(results['iterations']) // 100)
                x_values = results['iterations'][::subsample_rate]
                y_values = model_data[metric_key][::subsample_rate]
                
                plt.plot(
                    x_values, 
                    y_values, 
                    label=model_data['name'],
                    linestyle='--',
                    color='black',
                    linewidth=2.5,
                    alpha=0.7
                )
        
        for model_key, model_data in constrained_models.items():
            metric_key = f'online_{metric_name}'
            
            if metric_key in model_data and len(model_data[metric_key]) > 0:
                subsample_rate = max(1, len(results['iterations']) // 100)
                x_values = results['iterations'][::subsample_rate]
                y_values = model_data[metric_key][::subsample_rate]
                
                linestyle = '-'
                if 'rho=' in model_data['name']:
                    linestyle = ':'
                
                plt.plot(
                    x_values, 
                    y_values, 
                    label=model_data['name'],
                    linestyle=linestyle,
                    color=colors[color_idx % len(colors)],
                    linewidth=2,
                    alpha=0.8
                )
                color_idx += 1
        
        plt.title(f'Online {metric_name.capitalize()} - Dataset: {dataset_name}')
        plt.xlabel('Iterations (samples seen)')
        plt.ylabel(f'{metric_name.capitalize()}')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"combined_{metric_name}_{dataset_name}.png"))
        plt.close()
    
    plt.figure(figsize=(16, 10))
    
    metrics_for_comparison = ['accuracy', 'macrof1']
    model_keys = list(results['models'].keys())
    model_names = [results['models'][k]['name'] for k in model_keys]
    
    accuracy_values = []
    for k in model_keys:
        acc_key = 'online_accuracy'
        if acc_key in results['models'][k] and len(results['models'][k][acc_key]) > 0:
            accuracy_values.append(results['models'][k][acc_key][-1])
        else:
            accuracy_values.append(0)
    
    sorted_indices = np.argsort(accuracy_values)[::-1]  # Descending order
    model_keys = [model_keys[i] for i in sorted_indices]
    model_names = [model_names[i] for i in sorted_indices]
    
    bar_width = 0.35
    x = np.arange(len(model_keys))
    
    for i, metric_name in enumerate(metrics_for_comparison):
        metric_key = f'online_{metric_name}'
        values = []
        
        for k in model_keys:
            if metric_key in results['models'][k] and len(results['models'][k][metric_key]) > 0:
                values.append(results['models'][k][metric_key][-1])
            else:
                values.append(0)
        
        offset = i * bar_width - (len(metrics_for_comparison) * bar_width / 2) + bar_width/2
        plt.bar(x + offset, values, bar_width, label=metric_name.capitalize())
    
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.title(f'Final Metrics Comparison - {dataset_name}')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"combined_final_metrics_{dataset_name}.png"), bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(14, 6))
    
    model_leaves = []
    for k in model_keys:
        if 'active_leaves' in results['models'][k]:
            model_leaves.append(results['models'][k]['active_leaves'])
        else:
            model_leaves.append(0)
    
    bars = plt.bar(x, model_leaves, width=0.6, color=plt.cm.viridis(np.linspace(0, 0.8, len(model_keys))))
    
    for j, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f'{height}',
            ha='center', 
            va='bottom',
            fontsize=9
        )
    
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.title(f'Active Leaves Comparison - {dataset_name}')
    plt.ylabel('Number of Active Leaves')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"combined_model_size_{dataset_name}.png"), bbox_inches='tight')
    plt.close()

def create_summary_across_datasets(all_results, output_dir):
    """
    Create a summary of results across all datasets for Hoeffding Tree experiments.
    
    Parameters:
    -----------
    all_results : Dict[str, Dict]
        Dictionary mapping dataset names to combined results
    output_dir : str
        Directory to save summary
    """
    if not all_results:
        print("No results to summarize")
        return
    
    rows = []
    metrics_of_interest = ['accuracy', 'macrof1']
    
    experiment_types = [
        'No_Constraints',
        'All_Constraints',
        'Positive_Constraints_Only',
        'Negative_Constraints_Only'
    ]
    
    for dataset_name, results in all_results.items():
        for exp_type in experiment_types:
            matching_models = []
            for model_key, model_data in results['models'].items():
                if exp_type in model_key.replace(' ', '_'):
                    matching_models.append((model_key, model_data))
            
            if matching_models:
                model_key, model_data = matching_models[0]
                
                row = {'dataset': dataset_name, 'experiment_type': exp_type.replace('_', ' ')}
                
                for metric_name in metrics_of_interest:
                    metric_key = f'online_{metric_name}'
                    if metric_key in model_data and len(model_data[metric_key]) > 0:
                        row[metric_name] = model_data[metric_key][-1]
                
                if 'active_leaves' in model_data:
                    row['active_leaves'] = model_data['active_leaves']
                
                rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    summary_path = os.path.join(output_dir, "hoeffding_tree_experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    
    plt.figure(figsize=(18, 12))
    
    datasets = summary_df['dataset'].unique()
    exp_types = summary_df['experiment_type'].unique()
    
    x = np.arange(len(datasets))
    width = 0.8 / len(exp_types)
    
    for i, exp_type in enumerate(exp_types):
        exp_data = summary_df[summary_df['experiment_type'] == exp_type]
        accuracies = []
        
        for dataset in datasets:
            dataset_row = exp_data[exp_data['dataset'] == dataset]
            if not dataset_row.empty and 'accuracy' in dataset_row.columns:
                accuracies.append(dataset_row['accuracy'].iloc[0])
            else:
                accuracies.append(0)
        
        plt.bar(
            x + i * width - (len(exp_types) * width / 2) + width/2, 
            accuracies, 
            width, 
            label=exp_type
        )
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Datasets and Constraint Configurations')
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hoeffding_tree_accuracy_comparison.png"), bbox_inches='tight')
    plt.close()
    
    if 'active_leaves' in summary_df.columns:
        plt.figure(figsize=(18, 12))
        
        for i, exp_type in enumerate(exp_types):
            exp_data = summary_df[summary_df['experiment_type'] == exp_type]
            leaves = []
            
            for dataset in datasets:
                dataset_row = exp_data[exp_data['dataset'] == dataset]
                if not dataset_row.empty and 'active_leaves' in dataset_row.columns:
                    leaves.append(dataset_row['active_leaves'].iloc[0])
                else:
                    leaves.append(0)
            
            plt.bar(
                x + i * width - (len(exp_types) * width / 2) + width/2, 
                leaves, 
                width, 
                label=exp_type
            )
        
        plt.xlabel('Dataset')
        plt.ylabel('Active Leaves')
        plt.title('Model Size Comparison Across Datasets')
        plt.xticks(x, datasets, rotation=45, ha='right')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hoeffding_tree_model_size_comparison.png"), bbox_inches='tight')
        plt.close()

def run_all_experiments(data_dir, 
                       constraints_file=None,
                       max_datasets=None,
                       dataset_filter=None,
                       rho=1e-5,
                       stdevmul=2.0,
                       grace_period=200,
                       max_depth=None,
                       normalize=True,
                       output_dir="hoeffding_tree_results",
                       max_correction_factor=10000):
    """
    Run experiments on multiple UCI datasets using HoeffdingTreeClassifierMono.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing UCI datasets
    constraints_file : str, optional
        Path to a CSV file containing constraint definitions
    max_datasets : int, optional
        Maximum number of datasets to load
    dataset_filter : List[str], optional
        List of dataset names to load (if None, load all)
    rho : float
        The strength of the monotonicity penalty
    stdevmul : float
        Standard deviation multiplier for constraint checking
    grace_period : int
        Number of instances a leaf should observe between split attempts
    max_depth : int, optional
        The maximum depth the tree can reach
    normalize : bool
        Whether to normalize numeric features
    output_dir : str
        Directory to save results
    """
    loader = UCIDatasetLoader(data_dir, constraints_file)
    
    if constraints_file is None:
        constraints_file = os.path.join(data_dir, "constraints.csv")
        if not os.path.exists(constraints_file):
            loader.create_constraints_csv(constraints_file)
    
    datasets = loader.load_datasets(max_datasets=max_datasets, dataset_filter=dataset_filter)
    
    print(f"Loaded {len(datasets)} datasets")
    
    all_results = {}
    for name, (df, constraints) in datasets.items():
        print(f"\nProcessing dataset: {name}")
        
        results = run_multiple_experiments(
            dataset_name=name,
            df=df,
            constraints=constraints,
            rho=rho,
            stdevmul=stdevmul,
            grace_period=grace_period,
            max_depth=max_depth,
            normalize=normalize,
            output_dir=output_dir,
            max_correction_factor=max_correction_factor
        )
        
        if results:
            all_results[name] = results
    
    create_summary_across_datasets(all_results, output_dir)
    
    return all_results

def main():
    """
    Main function with argument parsing.
    """
    parser = argparse.ArgumentParser(description='Run Hoeffding Tree monotonicity constraint experiments')
    parser.add_argument('--data_dir', type=str, default='D:/mgr/mgrdeg/project/masters/data_dump',
                       help='Directory containing UCI datasets')
    parser.add_argument('--constraints_file', type=str, default=None,
                       help='Path to constraints CSV file')
    parser.add_argument('--max_datasets', type=int, default=None,
                       help='Maximum number of datasets to process')
    parser.add_argument('--dataset_filter', type=str, nargs='+', default=None,
                       help='Only process datasets matching these names')
    parser.add_argument('--rho', type=float, default=1e-5,
                       help='Monotonicity penalty parameter')
    parser.add_argument('--stdevmul', type=float, default=2.0,
                       help='Standard deviation multiplier')
    parser.add_argument('--grace_period', type=int, default=200,
                       help='Grace period between split attempts')
    parser.add_argument('--max_depth', type=int, default=None,
                       help='Maximum tree depth')
    parser.add_argument('--no_normalization', action='store_true',
                       help='Disable feature normalization')
    parser.add_argument('--output_dir', type=str, default='hoeffding_tree_results',
                       help='Directory to save results')
    parser.add_argument('--max_correction_factor', type=float, default=10000,
                       help='Maximum correction factor for Hoeffding Tree')
    
    args = parser.parse_args()
    
    run_all_experiments(
        data_dir=args.data_dir,
        constraints_file=args.constraints_file,
        max_datasets=args.max_datasets,
        dataset_filter=args.dataset_filter,
        rho=args.rho,
        stdevmul=args.stdevmul,
        grace_period=args.grace_period,
        max_depth=args.max_depth,
        normalize=not args.no_normalization,
        output_dir=args.output_dir,
        max_correction_factor=args.max_correction_factor
    )

if __name__ == "__main__":
    main()