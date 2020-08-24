import random
from typing import List

from nca.core.agent.agents import Agents
from nca.core.agent.individual import Individual
from nca.core.evolution.selection.selection import SurvivorSelection


class FitnessSteadyStateSelection(SurvivorSelection):

    def __init__(self):
        super().__init__()

    def __call__(self, individuals: List[Individual]) -> List[Individual]:
        return sorted(individuals, key=lambda x: x.fitness, reverse=True)


class GenerationalSteadyStateSelection(SurvivorSelection):

    def __init__(self):
        super().__init__()

    def __call__(self, individuals: List[Individual]) -> List[Individual]:
        return sorted(individuals, key=lambda x: x.age.generations, reverse=False)


class ElitismSelection(SurvivorSelection):

    def __init__(self):
        super().__init__()

    def __call__(self, individuals: List[Individual]) -> List[Individual]:
        sorted_individuals: List[Individual] = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        elite_individuals: List[Individual] = sorted_individuals[:self.configuration.selection_size]
        non_elite_individuals: List[Individual] = sorted_individuals[self.configuration.selection_size:]

        new_individuals = elite_individuals
        random.shuffle(non_elite_individuals)
        new_individuals.extend(non_elite_individuals)
        return new_individuals


class RoundRobinTournament(SurvivorSelection):

    def __init__(self):
        super().__init__()

    def __call__(self, individuals: Agents) -> List[Individual]:
        raise Exception("Unimplemented Round Robin Tournament")
        return []


class MuLambdaSelection(SurvivorSelection):

    def __init__(self):
        super().__init__()

    def __call__(self, individuals: Agents) -> List[Individual]:
        raise Exception("Unimplemented Mu Lambda Selection")
        return []


class NullSurvivorSelection(FitnessSteadyStateSelection):
    pass
