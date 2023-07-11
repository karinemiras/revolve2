"""Contains EnvironmentActorController, an environment controller for an environment with a single actor that uses a provided ActorController."""

from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, EnvironmentController


class EnvironmentActorController(EnvironmentController):
    """An environment controller for an environment with a single actor that uses a provided ActorController."""

    actor_controller: ActorController

    def __init__(self, actor_controller: ActorController) -> None:
        """
        Initialize this object.

        :param actor_controller: The actor controller to use for the single actor in the environment.
        """
        self.actor_controller = actor_controller

    def control(self, dt: float, actor_control: ActorControl, loop, results, joints_off) -> None:
        """
        Control the single actor in the environment using an ActorController.

        :param dt: Time since last call to this function.
        :param actor_control: Object used to interface with the environment.
        """
        if loop == 'closed':
            self.actor_controller.set_sensors(results)

        self.actor_controller.step(dt)
        dof_targets = self.actor_controller.get_dof_targets()

        if len(joints_off) > 0:
            for j in joints_off:
                dof_targets[j] = 0

        actor_control.set_dof_targets(0, dof_targets)
