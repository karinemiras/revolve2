import math
from sklearn.neighbors import KDTree
from typing import List, Optional, Tuple
import copy
import pprint

# TODO: because of the bizarre logic that ties gen0 to gen1, individuals from gen0 have incorrect relative measures
# it seems as if the measures of gen1 are correct, and get copied to gen0

# relative measures (which depend on the rest on the pop (or gens) to be calculated), and thus can change at every gen
# 'pool' measures depend on the pool of competitors and while 'pop' measures depends only on the survivors
# time dependent measures are also considered relative, e.g., age is relative to gens
class MeasureRelative:

    def __init__(self, genotype_measures=None, neighbours_measures=None):
        self._genotype_measures = genotype_measures
        self._neighbours_measures = neighbours_measures

    def _return_only_relative(self):
        # pool measures used for parent selection are not being saved.
        # they are overwritten, and only the values of survival selection are persisted.
        # persisted it in the future?
        relative_measures = ['pop_diversity',
                             'pool_diversity',
                             'pool_dominated_individuals',
                             'pool_fulldominated_individuals',
                             'age',
                             'inverse_age']

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

    # counts how many individuals of the current pool this individual dominates
    # an individual a dominates an individual b if a is better in at least one measure and not worse in any measure
    # better=higher > maximization
    def _pool_dominated_individuals(self):

        # TODO: make this a param in the exp manager
        which_measures = ['relative_displacement_y',
                          'inverse_age']

        pool_dominated_individuals = 0
        pool_fulldominated_individuals = 0
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
            if better == len(which_measures):
                pool_fulldominated_individuals += 1

        self._genotype_measures['pool_dominated_individuals'] = pool_dominated_individuals
        self._genotype_measures['pool_fulldominated_individuals'] = pool_fulldominated_individuals
        return self._genotype_measures

    def _age(self, generation_index):

        age = generation_index - self._genotype_measures['birth'] + 1
        inverse_age = 1/age
        self._genotype_measures['age'] = age
        self._genotype_measures['inverse_age'] = inverse_age

        return self._genotype_measures
