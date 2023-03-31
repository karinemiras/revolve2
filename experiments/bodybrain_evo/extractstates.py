
from pyrr import Quaternion, Vector3

from revolve2.core.physics.running import (
    ActorState,
)
def extracts_states(states):
    for idx_genotype in range(0, len(states.environment_results)):

        for idx_state in range(0, len(states.environment_results[idx_genotype].environment_states)):
            for idx_actor in range(0, len(
                    states.environment_results[idx_genotype].environment_states[idx_state].actor_states)):
                pose = states.environment_results[idx_genotype].environment_states[idx_state].actor_states[idx_actor][
                    "pose"]
                position = pose["p"][0]  # [0] is center of root element
                orientation = pose["r"][0]
                states.environment_results[idx_genotype].environment_states[idx_state].actor_states[idx_actor] = \
                    ActorState(
                        Vector3([position[0], position[1], position[2]]),
                        Quaternion(
                            [
                                orientation[0],
                                orientation[1],
                                orientation[2],
                                orientation[3],
                            ]
                        ),
                    )
    return states
