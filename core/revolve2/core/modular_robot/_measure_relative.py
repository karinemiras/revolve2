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


# relative measures (which depend on the rest on the pop to be calculated)
class MeasureRelative:

    _states: List[Tuple[float, RunnerState]]

    def __init__(self, genotype_measures=None, neighbours_measures=None):
        self._genotype_measures = genotype_measures
        self._neighbours_measures = neighbours_measures

    def _return_only_relative(self):
        relative_measures = ['diversity',
                             'dominated_individuals']

        copy_genotype_measures = copy.deepcopy(self._genotype_measures)

        for measure in self._genotype_measures:
            if measure not in relative_measures:
                del copy_genotype_measures[measure]

        return copy_genotype_measures

    def _diversity(self):

        # TODO: make this a param in the exp manager
        which_measures = ['symmetry',
                          'proportion',
                          'coverage',
                          'extremities_prop',
                          'hinge_prop',
                          'branching_prop']

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

        self._genotype_measures['diversity'] = diversity

        return self._genotype_measures

    # counts how many individuals of the current pop this individual dominates
    # an individual a dominates an individual b if a is better in at least one measure and not worse in any measure
    # TODO: allow the use
    def _dominated_individuals(self):

        # TODO: make this a param in the exp manager
        which_measures = ['symmetry',
                          'proportion']
        # print('>>>------')
        # pprint.pprint(self._genotype_measures)
        # print('------')
        # for neighbour_measures in self._neighbours_measures:
        #     pprint.pprint(neighbour_measures)
        #     better = 0
        #     worse = 0
        #     for key in which_measures:
        #         print(key, neighbour_measures[key], self._genotype_measures[key])
        #
        dominated_individuals = 0
        self._genotype_measures['dominated_individuals'] = dominated_individuals
        return self._genotype_measures
