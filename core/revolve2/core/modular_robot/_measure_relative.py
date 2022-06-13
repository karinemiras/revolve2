import math
from sklearn.neighbors import KDTree
from typing import List, Optional, Tuple
import copy
import pprint

from ._module import Module
from revolve2.core.physics.running import (
    RunnerState,
    ActorState
)


# relative measures (which depend on the rest on the pop (or gens) to be calculated), and thus can change at every gen
# 'pool' measures depend on the pool of competitors and while 'pop' measures depends only on the survivors
# time dependent measures are also considered relative, e.g., age is relative to gens
class MeasureRelative:

    _states: List[Tuple[float, RunnerState]]

    def __init__(self, genotype_measures=None, neighbours_measures=None):
        self._genotype_measures = genotype_measures
        self._neighbours_measures = neighbours_measures

    def _return_only_relative(self):
        relative_measures = ['pop_diversity',
                             'pool_diversity',
                             'pool_dominated_individuals',
                             'age']

        copy_genotype_measures = copy.deepcopy(self._genotype_measures)

        for measure in self._genotype_measures:
            if measure not in relative_measures:
                del copy_genotype_measures[measure]

        return copy_genotype_measures

    def _diversity(self, type='pop'):

        # TODO: make this a param in the exp manager
        which_measures = ['symmetry',
                          'proportion',
                          'coverage',
                          'extremities_prop',
                          'hinge_prop',
                          'branching_prop']
        # TODO: create age measure
        genotype_measures = []
        for key in which_measures:
            genotype_measures.append(self._genotype_measures[key])

        neighbours_measures = []
        for neighbour_measures in self._neighbours_measures:

            neighbours_measures.append([])
            for key in which_measures:
                neighbours_measures[-1].append(neighbour_measures[key])

        kdt = KDTree(neighbours_measures, leaf_size=30, metric='euclidean')

        # distances from neighbors
        distances, indexes = kdt.query([genotype_measures], k=len(self._neighbours_measures))
        diversity = sum(distances[0])/len(distances[0])

        self._genotype_measures[f'{type}_diversity'] = diversity

        return self._genotype_measures

    # counts how many individuals of the current pop this individual dominates
    # an individual a dominates an individual b if a is better in at least one measure and not worse in any measure
    # better=higher > maximization
    def _pool_dominated_individuals(self):

        # TODO: make this a param in the exp manager
        which_measures = ['displacement_xy',
                          'modules_count']

        pool_dominated_individuals = 0
        for neighbour_measures in self._neighbours_measures:
            better = 0
            worse = 0
            for key in which_measures:
                if self._genotype_measures[key] > neighbour_measures[key]:
                    better += 1
                if self._genotype_measures[key] < neighbour_measures[key]:
                    worse += 1
            if better > 0 and worse == 0:
                pool_dominated_individuals += 1

        self._genotype_measures['pool_dominated_individuals'] = pool_dominated_individuals
        return self._genotype_measures

    # TODO: assumes that first gen is as large as offspring size: adapt it for otherwise
    def _age(self, individual_id, offspring_size):
        age = math.floor(float(individual_id) / offspring_size)
        self._genotype_measures['age'] = age
        return self._genotype_measures