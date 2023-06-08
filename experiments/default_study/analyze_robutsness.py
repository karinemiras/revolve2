"""
Visualize and run a modular robot using Mujoco.
"""

from pyrr import Quaternion, Vector3
import argparse
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, Core, Module
from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from  sqlalchemy.sql.expression import func
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer, DbEnvconditions
from genotype import GenotypeSerializer, develop
from optimizer import DbOptimizerState
import sys, os, copy
from revolve2.core.modular_robot.render.render import Render
from revolve2.core.modular_robot import Measure
from revolve2.core.database.serializers import DbFloat
import pprint
import random
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
        parser.add_argument("bisymmetry")

        args = parser.parse_args()

        self.study = args.study
        self.experiments_name = ['qbilateral'] # args.experiments.split(',')
        self.runs = [2] #args.watchruns.split(',')
        mainpath = args.mainpath
        self.simulator = args.simulator
        self.loop = args.loop
        self.body_phenotype = args.body_phenotype
        self.bisymmetry = list(map(int, args.bisymmetry.split(',')))

        self.generations = [0,50,100,150,200] 
        self.type_damage = ['body']#, 'brain']
        self.damages = [0]#[0, 0.2, 0.3, 0.4]
        self.sample_size = 5

        self.path = f'{mainpath}/{self.study}'
        self.anal_path = f'{self.path}/analysis/damage'
        if not os.path.exists(self.anal_path):
            os.makedirs(self.anal_path)

        if self.simulator == 'mujoco':
            self._TERRAIN = terrains.flat()

        for ids, experiment_name in enumerate(self.experiments_name):
            print('\n', experiment_name)
            for run in self.runs:
                print('\n run: ', run)
                db = open_async_database_sqlite(f'{self.path}/{experiment_name}/run_{run}')

                for gen in self.generations:
                    print('  in gen: ', gen)
                    await self.recover(db, experiment_name, run, gen, ids)


    async def recover(self, db, experiment_name, run, gen, ids):
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

            query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat) \
                .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                        & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                        & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                        & DbEAOptimizerGeneration.generation_index.in_([gen])
                        )
            query = query.order_by(func.random())
            # DbFloat.speed_y.desc())


            # TODO: all or 10 best?
            rows = ((await session.execute(query)).all())
            for idx, r in enumerate(rows[0:self.sample_size]):
                env_conditions_id = r.DbEAOptimizerGeneration.env_conditions_id
                print(f'  id:{r.DbEAOptimizerIndividual.individual_id} ' \
                                                f' birth:{r.DbFloat.birth} ' \
                         f' cond:{env_conditions_id} ' \
                         f' dom:{r.DbEAOptimizerGeneration.seasonal_dominated} ' \
                         f' speed_y:{r.DbFloat.speed_y}' \
                      )

                genotype = (
                    await GenotypeSerializer.from_database(
                        session, [r.DbEAOptimizerIndividual.genotype_id]
                    )
                )[0]

                for damage_type in self.type_damage:
                    for damage in self.damages:

                        phenotype, queried_substrate = develop(genotype, genotype.mapping_seed, max_modules,
                                                               substrate_radius, env_conditions[env_conditions_id],
                                                               len(env_conditions), plastic_body, plastic_brain,
                                                               self.loop, self.body_phenotype, self.bisymmetry[ids])

                        render = Render()
                        img_path = f'{self.anal_path}/' \
                                   f'{experiment_name}_{run}_{gen}_{r.DbEAOptimizerIndividual.genotype_id}_{damage}.png'

                        if damage > 0:
                            if damage_type == "body":
                                await self.damage_body(queried_substrate, damage)
                            else:
                                await self.damage_brain(queried_substrate, damage)

                        # export images of only a couple of examples
                       # if idx <= 2:
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
                                Vector3(
                                    [
                                        0.0,
                                        0.0,
                                        (bounding_box.size.z / 2.0 - bounding_box.offset.z),
                                    ]
                                ),
                                Quaternion.from_eulers([robot_rotation, 0, 0]),
                                [0.0 for _ in controller.get_dof_targets()],
                            )
                        )

                        batch = Batch(
                             simulation_time=simulation_time,
                             sampling_frequency=sampling_frequency,
                             control_frequency=control_frequency,
                         )
                        batch.environments.append(env)

                        # if self.simulator == 'isaac':
                        #     runner = LocalRunnerI(
                        #         headless=False,
                        #         env_conditions=env_conditions[env_conditions_id],
                        #         real_time=False,
                        #         loop=self.loop)
                        #
                        # elif self.simulator == 'mujoco':
                        #     runner = LocalRunnerM(headless=False, loop=self.loop)
                        #
                        # states = await runner.run_batch(batch)
                        # if self.simulator == 'isaac':
                        #     states = extracts_states(states)
                        #
                        # m = Measure(states=states, genotype_idx=0, phenotype=phenotype,
                        #             generation=0, simulation_time=simulation_time)
                       # pprint.pprint(m.measure_all_non_relative())


    async def damage_body(self, queried_substrate, damage):
        # removes random extremities

        modules_available = len(queried_substrate)-1
        to_remove = int(modules_available * damage)
        removed = 0
        l = list(queried_substrate.items())
        random.shuffle(l)
        queried_substrate = dict(l)

        while removed < to_remove:
            to_pop = []
            for key in queried_substrate:
                remove = False
                if (type(queried_substrate[key]) == ActiveHinge
                        and queried_substrate[key].children[ActiveHinge.ATTACHMENT] == None) \
                    or ( type(queried_substrate[key]) != ActiveHinge \
                        and queried_substrate[key].children[Core.FRONT] == None \
                        and queried_substrate[key].children[Core.RIGHT] == None \
                        and queried_substrate[key].children[Core.LEFT] == None):
                   remove = True

                if remove:
                    if removed < to_remove:
                        queried_substrate[key]._parent.children[queried_substrate[key].direction_from_parent] = None
                        to_pop.append(key)
                        removed += 1
                    else:
                        break
            for key in to_pop:
                queried_substrate.pop(key)

async def main() -> None:

    sim = Simulator()
    await sim.simulate()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



