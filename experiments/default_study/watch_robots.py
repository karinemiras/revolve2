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
import sys
from revolve2.core.modular_robot.render.render import Render


if sys.platform == "linux" or sys.platform == "linux2":
    from revolve2.runners.isaacgym import LocalRunner
else:
    from revolve2.runners.mujoco import LocalRunner


class Simulator:
    _controller: ActorController

    async def simulate(self) -> None:



        study = 'default_study'
        experiments_name = ['speed']
        runs = [3] #list(range(1, 11))
        generations = [100]
        bests = 5

        for experiment_name in experiments_name:
            print(experiment_name)
            for run in runs:
                print(' run: ', run)

                if sys.platform == "linux" or sys.platform == "linux2":
                    path = f'/storage/karine/{study}'
                else:
                    path = f'/Users/karinemiras/Documents/storage_ripper2/{study}'

                db = open_async_database_sqlite(f'{path}/{experiment_name}/run_{run}')

                for gen in generations:
                    print('  gen: ', gen)

                    async with AsyncSession(db) as session:

                        rows = (
                            (await session.execute(select(DbEAOptimizer))).all()
                        )
                        max_modules = rows[0].DbEAOptimizer.max_modules
                        substrate_radius = rows[0].DbEAOptimizer.substrate_radius

                        rows = (
                            (await session.execute(select(DbEAOptimizerGeneration, DbEAOptimizerIndividual)
                                                   .filter(DbEAOptimizerGeneration.generation_index.in_([gen]))
                                                   .filter(
                                DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                                   .order_by(
                                DbEAOptimizerGeneration.pool_dominated_individuals.desc())

                                                   )).all()
                        )

                        for idx, r in enumerate(rows[0:bests]):
                            print(f'rank:{idx} id:{r.DbEAOptimizerIndividual.individual_id} dom:{r.DbEAOptimizerGeneration.pool_dominated_individuals}')
                            genotype = (
                                await GenotypeSerializer.from_database(
                                    session, [r.DbEAOptimizerIndividual.genotype_id]
                                )
                            )[0]

                            phenotype = develop(genotype, genotype.mapping_seed, max_modules, substrate_radius)
                            render = Render()
                            img_path = f'{r.DbEAOptimizerIndividual.individual_id}.png'
                            render.render_robot(phenotype.body.core, img_path)

                            actor, self._controller = phenotype.make_actor_and_controller()

                            env = Environment()
                            env.actors.append(
                                PosedActor(
                                    actor,
                                    Vector3([0.0, 0.0, 0.1]),
                                    Quaternion(),
                                    [0.0 for _ in self._controller.get_dof_targets()],
                                )
                            )
                            # TODO : make a loop?
                            batch = Batch(
                                simulation_time=5,
                                sampling_frequency=0.0001,
                                control_frequency=10,
                                control=self._control,
                            )
                            batch.environments.append(env)
                            if sys.platform == "linux" or sys.platform == "linux2":
                                runner = LocalRunner(LocalRunner.SimParams())
                            else:
                                runner = LocalRunner()
                            await runner.run_batch(batch)
                            print('domne')

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
