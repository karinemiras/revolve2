"""
Visualize and run a modular robot using Mujoco.
"""

from pyrr import Quaternion, Vector3

from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor

from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer
from genotype import GenotypeSerializer, develop
from optimizer import DbOptimizerState
import sys
from revolve2.core.modular_robot.render.render import Render
from revolve2.core.modular_robot import Measure
from revolve2.core.database.serializers import DbFloat
import pprint
import numpy

if sys.platform == "linux" or sys.platform == "linux2":
    from revolve2.runners.isaacgym import LocalRunner
else:
    from revolve2.runners.mujoco import LocalRunner


class Simulator:
    _controller: ActorController

    async def simulate(self) -> None:

        # nice one: speed20_10
        self.study = 'default_study'
        self.experiments_name = ['speed20']
        self.runs = [2]#list(range(1, 20+1))
        self.generations = [200]
        self.bests = 1
        self.specific_robot = 1
        # 'all' selects best from the whole experiment
        # 'gens' selects best from chosen generations
        # 'specific' selects teh robot provided in specific_robot
        self.bests_type = 'all'

        for experiment_name in self.experiments_name:
            print('\n', experiment_name)
            for run in self.runs:
                print('\n run: ', run)

                if sys.platform == "linux" or sys.platform == "linux2":
                    path = f'/storage/karine/{self.study}'
                else:
                    path = f'/Users/karinemiras/Documents/storage_ripper2/{self.study}'

                db = open_async_database_sqlite(f'{path}/{experiment_name}/run_{run}')

                if self.bests_type == 'gens':
                    for gen in self.generations:
                        print('  in gen: ', gen)
                        await self.recover(db, gen, path)
                elif self.bests_type == 'all':
                    print('  within all gens')
                    await self.recover(db, -1, path)
                elif self.bests_type == 'specific':
                    await self.recover(db, -1, path)

    async def recover(self, db, gen, path):
        async with AsyncSession(db) as session:

            rows = (
                (await session.execute(select(DbEAOptimizer))).all()
            )
            max_modules = rows[0].DbEAOptimizer.max_modules
            substrate_radius = rows[0].DbEAOptimizer.substrate_radius

            rows = (
                (await session.execute(select(DbOptimizerState))).all()
            )
            sampling_frequency = rows[0].DbOptimizerState.sampling_frequency
            control_frequency = rows[0].DbOptimizerState.control_frequency
            simulation_time = rows[0].DbOptimizerState.simulation_time

            # TODO: make queries more reusable...
            if gen == -1:
                if self.bests_type == 'specific':
                    rows = ((await session.execute(select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat)
                                               .filter(  (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                               & (DbEAOptimizerIndividual.individual_id == self.specific_robot)) )).all())
                else:
                    rows = ((await session.execute(select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat)
                                               .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                               & (DbFloat.id == DbEAOptimizerIndividual.float_id)).order_by(DbFloat.speed_x.desc()) )).all())
            else:
                rows = ((await session.execute(select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat)
                                           .filter(DbEAOptimizerGeneration.generation_index.in_([gen]))
                                           .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                          & (DbFloat.id == DbEAOptimizerIndividual.float_id)).order_by(DbFloat.speed_x.desc()) )).all())

            for idx, r in enumerate(rows[0:self.bests]):
                print(f'\n    rank:{idx} id:{r.DbEAOptimizerIndividual.individual_id} ' \
                      f' birth:{r.DbFloat.birth} ' \
                      f' speed_x:{r.DbFloat.speed_x}' \
                      )
                genotype = (
                    await GenotypeSerializer.from_database(
                        session, [r.DbEAOptimizerIndividual.genotype_id]
                    )
                )[0]

                phenotype = develop(genotype, genotype.mapping_seed, max_modules, substrate_radius)
                render = Render()
                img_path = f'{path}/analysis/currentinsim.png'
                render.render_robot(phenotype.body.core, img_path)

                actor, self._controller = phenotype.make_actor_and_controller()
                bounding_box = actor.calc_aabb()

                env = Environment()
                env.actors.append(
                    PosedActor(
                        actor,
                        Vector3([0.0, 0.0,  bounding_box.size.z / 2.0 - bounding_box.offset.z,]),
                        Quaternion(),
                        [0.0 for _ in self._controller.get_dof_targets()],
                    )
                )

                states = None
                batch = Batch(
                     simulation_time=simulation_time,
                     sampling_frequency=sampling_frequency,
                     control_frequency=control_frequency,
                     control=self._control,
                 )
                batch.environments.append(env)
                if sys.platform == "linux" or sys.platform == "linux2":
                    # TODO: take values from database
                    static_friction = 1.0
                    dynamic_friction = 1.0
                    gravity = "0;0;-9.81"
                    y_rotation_degrees = "15"
                    env_conditions_plane = [static_friction, dynamic_friction, gravity, y_rotation_degrees]
                    env_conditions = [env_conditions_plane]
                    runner = LocalRunner(LocalRunner.SimParams(), env_conditions=env_conditions[0])
                else:
                    runner = LocalRunner()
                states = await runner.run_batch(batch)

                m = Measure(states=states, genotype_idx=0, phenotype=phenotype,
                            generation=0, simulation_time=simulation_time)
                pprint.pprint(m.measure_all_non_relative())

    if sys.platform == "linux" or sys.platform == "linux2":
        def _control(self, dt: float, control: ActorControl) -> None:
            self._controller.step(dt)
            control.set_dof_targets(0, 0, self._controller.get_dof_targets())
    else:
        # TODO: merge with development and remove this
        def _control(
            self, environment_index: int, dt: float, control: ActorControl
        ) -> None:
            self._controller.step(dt)
            control.set_dof_targets(0, self._controller.get_dof_targets())


async def main() -> None:

    sim = Simulator()
    await sim.simulate()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
