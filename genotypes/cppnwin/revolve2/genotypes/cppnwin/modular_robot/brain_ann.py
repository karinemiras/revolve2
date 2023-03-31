from typing import List, Set, Tuple, cast
import random
import multineat
from squaternion import Quaternion
import pprint
import numpy as np
import math

from revolve2.core.modular_robot import ActiveHinge, Body
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import (
    BatchResults,
)

class BrainANN(ActorController):
    _genotype: multineat.Genome

    def __init__(self, genotype: multineat.Genome):
        self._genotype = genotype
        self._dof_targets = []
        self._sensors = None
        self._n_joints = 0
        self._dof_ranges = 1
        self._steps = 0

    def set_sensors(self, sensors: BatchResults):
        self._sensors = sensors

    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        self._n_joints = len(dof_ids)
        self._dof_targets = [0] * self._n_joints

        return self

    def step(self, dt: float) -> None:
        brain_net = multineat.NeuralNetwork()
        self._genotype.BuildPhenotype(brain_net)

        sin = math.sin(self._steps)
        closed_loop = self._sensors + sin

        output = self._evaluate_network(
            brain_net, closed_loop,
        )

        self._dof_targets = list(np.clip(output, a_min=-self._dof_ranges, a_max=self._dof_ranges))
        self._steps += 1

    def get_dof_targets(self) -> List[float]:
        return self._dof_targets

    @staticmethod
    def _evaluate_network(
        network: multineat.NeuralNetwork, inputs: List[float]
    ) -> float:
        network.Input(inputs)
        network.ActivateAllLayers()
        return cast(float, network.Output())  # TODO missing multineat typing
    def serialize(self):
        pass

    @classmethod
    def deserialize(cls, data):
        pass