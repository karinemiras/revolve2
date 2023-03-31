"""
Visualize and run a modular robot using Isaac Gym.
"""

import math
from random import Random

from pyrr import Quaternion, Vector3

from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import BrainCpgNetworkNeighbourRandom
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from revolve2.runners.isaacgym import LocalRunner
from revolve2.core.modular_robot.render.render import Render

class Simulator:
    _controller: ActorController

    async def simulate(self, robot: ModularRobot, control_frequency: float) -> None:
        batch = Batch(
            simulation_time=1000000,
            sampling_frequency=5,
            control_frequency=control_frequency,
            control=self._control,
        )

        actor, self._controller = robot.make_actor_and_controller()

        env = Environment()
        env.actors.append(
            PosedActor(
                actor,
                Vector3([0.0, 0.0, 0.1]),
                Quaternion(),
                [0.0 for _ in self._controller.get_dof_targets()],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner(LocalRunner.SimParams(), env_conditions=[1.0, 1.0, 0, 0, 0])
        await runner.run_batch(batch)

    def _control(self, dt: float, control: ActorControl) -> None:
        self._controller.step(dt)
        control.set_dof_targets(0, 0, self._controller.get_dof_targets())


async def main() -> None:
    rng = Random()
    rng.seed(5)

    body = Body()
    body.core._id = 1

###
    body.core.front = ActiveHinge(math.pi / 2.0)
    body.core.front._id = 2
    body.core.front._absolute_rotation = 90

    body.core.front.attachment = Brick(0.0)
    body.core.front.attachment._id = 3
    body.core.front.attachment._absolute_rotation = 0

    body.core.front.attachment.front = ActiveHinge(math.pi / 2.0)
    body.core.front.attachment.front._id = 4
    body.core.front.attachment.front._absolute_rotation = 0

    body.core.front.attachment.front.attachment = Brick(0.0)
    body.core.front.attachment.front.attachment._id = 5
    body.core.front.attachment.front.attachment._absolute_rotation = 0

###
    body.core.right = ActiveHinge(math.pi / 2.0)
    body.core.right._id = 6
    body.core.right._absolute_rotation = 90

    body.core.right.attachment = Brick(0.0)
    body.core.right.attachment._id = 7
    body.core.right.attachment._absolute_rotation = 0

    body.core.right.attachment.front = ActiveHinge(math.pi / 2.0)
    body.core.right.attachment.front._id = 8
    body.core.right.attachment.front._absolute_rotation = 0

    body.core.right.attachment.front.attachment = Brick(0.0)
    body.core.right.attachment.front.attachment._id = 9
    body.core.right.attachment.front.attachment._absolute_rotation = 0

###
    body.core.back = ActiveHinge(math.pi / 2.0)
    body.core.back._id = 10
    body.core.back._absolute_rotation = 90

    body.core.back.attachment = Brick(0.0)
    body.core.back.attachment._id = 11
    body.core.back.attachment._absolute_rotation = 0

    body.core.back.attachment.front = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.front._id = 12
    body.core.back.attachment.front._absolute_rotation = 0

    body.core.back.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment._id = 13
    body.core.back.attachment.front.attachment._absolute_rotation = 0

###
    body.core.left = ActiveHinge(math.pi / 2.0)
    body.core.left._id = 14
    body.core.left._absolute_rotation = 90

    body.core.left.attachment = Brick(0.0)
    body.core.left.attachment._id = 15
    body.core.left.attachment._absolute_rotation = 0

    body.core.left.attachment.front = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment.front._id = 16
    body.core.left.attachment.front._absolute_rotation = 0

    body.core.left.attachment.front.attachment = Brick(0.0)
    body.core.left.attachment.front.attachment._id = 17
    body.core.left.attachment.front.attachment._absolute_rotation = 0

    body.finalize()

    brain = BrainCpgNetworkNeighbourRandom(rng)
    robot = ModularRobot(body, brain)

    render = Render()
    img_path = f'manualrobot.png'
    render.render_robot(robot.body.core, img_path)

    sim = Simulator()
    await sim.simulate(robot, 20)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
