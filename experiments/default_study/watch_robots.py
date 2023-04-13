"""
Visualize and run a modular robot using Mujoco.
"""

from pyrr import Quaternion, Vector3
import argparse
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor

from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer, DbEnvconditions
from genotype import GenotypeSerializer, develop
from optimizer import DbOptimizerState
import sys
from revolve2.core.modular_robot.render.render import Render
from revolve2.core.modular_robot import Measure
from revolve2.core.database.serializers import DbFloat
import pprint
import numpy as np
from ast import literal_eval
from revolve2.core.physics.environment_actor_controller import (
    EnvironmentActorController,
)

from revolve2.runners.mujoco import LocalRunner as LocalRunnerM
from revolve2.runners.isaacgym import LocalRunner  as LocalRunnerI

from extractstates import *
from body_spider import *

from revolve2.standard_resources import terrains


class Simulator:
    _controller: ActorController

    async def simulate(self) -> None:

        parser = argparse.ArgumentParser()
        parser.add_argument("study")
        parser.add_argument("experiments")
        parser.add_argument("watchruns")
        parser.add_argument("generations")
        parser.add_argument("mainpath")
        parser.add_argument("simulator")
        parser.add_argument("loop")
        parser.add_argument("body_phenotype")

        args = parser.parse_args()

        self.study = args.study
        self.experiments_name = args.experiments.split(',')
        self.runs = args.watchruns.split(',')
        self.generations = list(map(int, args.generations.split(',')))
        mainpath = args.mainpath
        self.simulator = args.simulator
        self.loop = args.loop
        self.body_phenotype = args.body_phenotype

        self.bests = 1
        # 'all' selects best from all individuals
        # 'gens' selects best from chosen generations
        self.bests_type = 'gens'

        if self.simulator == 'mujoco':
            self._TERRAIN = terrains.flat()

        for experiment_name in self.experiments_name:
            print('\n', experiment_name)
            for run in self.runs:
                print('\n run: ', run)

                path = f'{mainpath}/{self.study}'

                db = open_async_database_sqlite(f'{path}/{experiment_name}/run_{run}')

                if self.bests_type == 'gens':
                    for gen in self.generations:
                        print('  in gen: ', gen)
                        await self.recover(db, gen, path)
                elif self.bests_type == 'all':
                    pass
                    # TODO: implement

    async def recover(self, db, gen, path):
        async with AsyncSession(db) as session:

            rows = (
                (await session.execute(select(DbEAOptimizer))).all()
            )
            max_modules = rows[0].DbEAOptimizer.max_modules
            substrate_radius = rows[0].DbEAOptimizer.substrate_radius
            plastic_body = rows[0].DbEAOptimizer.plastic_body
            plastic_brain = rows[0].DbEAOptimizer.plastic_brain

            rows = (
                (await session.execute(select(DbOptimizerState))).all()
            )
            sampling_frequency = rows[0].DbOptimizerState.sampling_frequency
            control_frequency = rows[0].DbOptimizerState.control_frequency
            simulation_time = rows[0].DbOptimizerState.simulation_time

            rows = ((await session.execute(select(DbEnvconditions))).all())
            env_conditions = {}
            for c_row in rows:
                env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)

            if self.bests_type == 'all':
                pass

            elif self.bests_type == 'gens':
                query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat) \
                    .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                            & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                            & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                            & DbEAOptimizerGeneration.generation_index.in_([gen])
                            # IF YOU WANNA SEE A SPECIFIC ROBOT IN THA GEN
                            #     & (DbEAOptimizerIndividual.individual_id == 4910)
                            )

                # if seasonal setup, criteria is seasonal pareto
                if len(rows) > 1:
                    query = query.order_by(
                                           # CAN ALSO USE SOME OTHER CRITERIA INSTEAD OF SEASONAL
                                           DbEAOptimizerGeneration.seasonal_dominated.desc(),
                        
                                           DbEAOptimizerGeneration.individual_id.asc(),
                                           DbEAOptimizerGeneration.env_conditions_id.asc())
                else:
                    query = query.order_by(DbFloat.speed_y.desc())

                rows = ((await session.execute(query)).all())

                num_lines = self.bests * len(env_conditions)
                for idx, r in enumerate(rows[0:num_lines]):
                    env_conditions_id = r.DbEAOptimizerGeneration.env_conditions_id
                    print(f'\n  id:{r.DbEAOptimizerIndividual.individual_id} ' \
                                                    f' birth:{r.DbFloat.birth} ' \
                             f' cond:{env_conditions_id} ' \
                             f' dom:{r.DbEAOptimizerGeneration.seasonal_dominated} ' \
                             f' speed_y:{r.DbFloat.speed_y} \n' \
                          )

                    genotype = (
                        await GenotypeSerializer.from_database(
                            session, [r.DbEAOptimizerIndividual.genotype_id]
                        )
                    )[0]

                    phenotype, queried_substrate = develop(genotype, genotype.mapping_seed, max_modules,
                                                           substrate_radius, env_conditions[env_conditions_id],
                                                            len(env_conditions), plastic_body, plastic_brain,
                                                            self.loop, self.body_phenotype )
                    render = Render()
                    img_path = f'{path}/currentinsim.png'

                    render.render_robot(phenotype.body.core, img_path)

                    actor, controller = phenotype.make_actor_and_controller()
                    bounding_box = actor.calc_aabb()
                    env = Environment(EnvironmentActorController(controller))

                    if self.simulator == 'mujoco':
                        env.static_geometries.extend(self._TERRAIN.static_geometry)

                    x_rotation_degrees = float(env_conditions[env_conditions_id][2])
                    robot_rotation = x_rotation_degrees * np.pi / 180

                    env.actors.append(
                        PosedActor(
                            actor,
                            Vector3([0.0, 0.0,  bounding_box.size.z / 2.0 - bounding_box.offset.z]),
                            Quaternion.from_eulers([robot_rotation, 0, 0]),
                            [0.0 for _ in controller.get_dof_targets()],
                        )
                    )

                    states = None
                    batch = Batch(
                         simulation_time=simulation_time,
                         sampling_frequency=sampling_frequency,
                         control_frequency=control_frequency,
                     )
                    batch.environments.append(env)

                    if self.simulator == 'isaac':
                        runner = LocalRunnerI(
                            headless=False,
                            env_conditions=env_conditions[env_conditions_id],
                            real_time=True,
                            loop=self.loop)

                    elif self.simulator == 'mujoco':
                        runner = LocalRunnerM(headless=False, loop=self.loop)

                    states = await runner.run_batch(batch)
                    if self.simulator == 'isaac':
                        states = extracts_states(states)

                    m = Measure(states=states, genotype_idx=0, phenotype=phenotype,
                                generation=0, simulation_time=simulation_time)
                    pprint.pprint(m.measure_all_non_relative())

async def main() -> None:

    sim = Simulator()
    await sim.simulate()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



