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
from genotype import GenotypeSerializer, develop_knockout
from optimizer import DbOptimizerState
import sys
from revolve2.core.modular_robot.render.render import Render
from revolve2.core.modular_robot import Measure
from revolve2.core.database.serializers import DbFloat
import pprint
import numpy as np
import os
from ast import literal_eval

from revolve2.runners.isaacgym import LocalRunner as LocalRunnerI


from body_spider import *

#/buildAgent/work/99bede84aa0a52c2/source/gpubroadphase/src/PxgAABBManager.cpp (1048) : invalid parameter : The application needs to increase PxgDynamicsMemoryConfig::foundLostAggregatePairsCapacity to 299692735 , otherwise, the simulation will miss interactions

class Simulator:
    _controller: ActorController

    async def simulate(self) -> None:

        parser = argparse.ArgumentParser()
        parser.add_argument("study")
        parser.add_argument("experiments")
        parser.add_argument("tfs")
        parser.add_argument("watchruns")
        parser.add_argument("generations")
        parser.add_argument("mainpath")

        args = parser.parse_args()

        self.study = args.study
        self.experiments_name = args.experiments.split(',')
        self.tfs = list(args.tfs.split(','))
        self.runs = args.watchruns.split(',')
        self.generations = [0, 25, 100] # list(map(int, args.generations.split(',')))
        test_robots = []
        self.mainpath = args.mainpath

        self.bests = 5
        # 'all' selects best from all individuals
        # 'gens' selects best from chosen generations
        self.bests_type = 'gens'
        self.ranking = ['best', 'worst']
        self.header = False

        path = f'{self.mainpath}/{self.study}/analysis/knockouts/'
        if not os.path.exists(path):
            os.makedirs(path)

        for ids, experiment_name in enumerate(self.experiments_name):
            print('\n',experiment_name)
            for run in self.runs:
                print('run: ', run)

                path = f'{self.mainpath}/{self.study}'

                fpath = f'{path}/{experiment_name}/run_{run}'
                db = open_async_database_sqlite(fpath)

                if self.bests_type == 'gens':
                    for gen in self.generations:
                        print('  gen: ', gen)
                        await self.recover(db, gen, path, test_robots, self.tfs[ids], experiment_name, run)
                elif self.bests_type == 'all':
                    pass
                    # TODO: implement

    async def recover(self, db, gen, path, test_robots, tfs, experiment_name, run):
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

                for ranking in self.ranking:

                    query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat) \
                        .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                                & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                & DbEAOptimizerGeneration.generation_index.in_([gen])
                                )

                    if len(test_robots) > 0:
                        query = query.filter(DbEAOptimizerIndividual.individual_id.in_(test_robots))

                    print(' ', ranking)
                    if ranking == 'best':
                        # if seasonal setup, criteria is seasonal pareto
                        if len(rows) > 1:
                            query = query.order_by(
                                                   # CAN ALSO USE SOME OTHER CRITERIA INSTEAD OF SEASONAL
                                                   DbEAOptimizerGeneration.seasonal_dominated.desc(),
                                                   DbEAOptimizerGeneration.individual_id.asc(),
                                                   DbEAOptimizerGeneration.env_conditions_id.asc())
                        else:
                            query = query.order_by(DbFloat.speed_y.desc())
                    else:
                        if len(rows) > 1:
                            query = query.order_by(
                                                   DbEAOptimizerGeneration.seasonal_dominated.asc(),
                                                   DbEAOptimizerGeneration.individual_id.asc(),
                                                   DbEAOptimizerGeneration.env_conditions_id.asc())
                        else:
                            query = query.order_by(DbFloat.speed_y.asc())

                    rows = ((await session.execute(query)).all())

                    num_lines = self.bests * len(env_conditions)
                    for idx, r in enumerate(rows[0:num_lines]):
                        env_conditions_id = r.DbEAOptimizerGeneration.env_conditions_id
                        # print(f'\n  rk:{idx+1} ' \
                        #          f' id:{r.DbEAOptimizerIndividual.individual_id} ' \
                        #          f' birth:{r.DbFloat.birth} ' \
                        #          f' gen:{r.DbEAOptimizerGeneration.generation_index} ' \
                        #          f' cond:{env_conditions_id} ' \
                        #          f' dom:{r.DbEAOptimizerGeneration.seasonal_dominated} ' \
                        #          f' speed_y:{r.DbFloat.speed_y} ' \
                        #       )

                        genotype = (
                            await GenotypeSerializer.from_database(
                                session, [r.DbEAOptimizerIndividual.genotype_id]
                            )
                        )[0]
                        geno_size = len(genotype.body.genotype)
                        knockout = None
                        knockstring = 'o'

                        original_phenotype, original_substrate, promotors = \
                            develop_knockout(genotype, genotype.mapping_seed, max_modules, tfs,
                                             substrate_radius, env_conditions[env_conditions_id],
                                             len(env_conditions), plastic_body, plastic_brain,
                                             knockout)

                        async def analyze(phenotype, experiment_name, run, gen, individual_id, knockout, geno_size, promotors, ranking, distance):
                            # render = Render()
                            # img_path = f'{self.mainpath}/{self.study}/analysis/knockouts/{experiment_name}_{run}_{gen}_{individual_id}_{knockout}.png'
                            # render.render_robot(phenotype.body.core, img_path)

                            actor,  self._controller = phenotype.make_actor_and_controller()
                            bounding_box = actor.calc_aabb()

                            env = Environment()
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
                                    [0.0 for _ in  self._controller.get_dof_targets()],
                                )
                            )

                            batch = Batch(
                                 simulation_time=simulation_time,
                                 sampling_frequency=sampling_frequency,
                                 control_frequency=control_frequency,
                                 control=self._control,
                             )
                            batch.environments.append(env)

                            runner = LocalRunnerI(LocalRunnerI.SimParams(),
                                headless=True,
                                env_conditions=env_conditions[env_conditions_id],
                                real_time=False,)
                            states = None
                            states = await runner.run_batch(batch)

                            m = Measure(states=states, genotype_idx=0, phenotype=phenotype,
                                        generation=0, simulation_time=simulation_time)
                            #pprint.pprint(m.measure_all_non_relative())
                            measures = m.measure_all_non_relative()

                            pfile = f'{self.mainpath}/{self.study}/analysis/knockouts/knockouts.csv'
                            if not self.header:
                                header = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id', 'knockout',
                                          'geno_size', 'promotors', 'distance'] + list(measures.keys())
                                with open(pfile, 'w') as file:
                                    file.write(','.join(map(str, header)))
                                    file.write('\n')
                                self.header = True

                            with open(pfile, 'a') as file:
                                file.write(','.join(map(str, [experiment_name, run, gen, ranking, individual_id,
                                                              knockout, geno_size, promotors, distance] +
                                                              list(measures.values())
                                                        )))
                                file.write('\n')

                        #print('\noriginal')
                        distance = 0
                        await analyze(original_phenotype, experiment_name, run, gen, r.DbEAOptimizerIndividual.individual_id,
                                      knockstring, geno_size, len(promotors), ranking, distance)

                        #print('\nknockouts')
                        singles = [[i] for i in range(len(promotors))]
                        pairs = [[singles[i][0], singles[i + 1][0]] for i in range(len(singles) - 1)]
                        pairs.append([singles[len(singles) - 1][0], singles[0][0]])
                        # every individual promoter and all sequential pairs
                        knockouts = singles + pairs

                        for knockout in knockouts:
                            #print(knockout)
                            knockstring = '.'.join([str(item) for item in knockout])
                            knockout_phenotype, knockout_substrate, promotors = \
                                develop_knockout(genotype, genotype.mapping_seed, max_modules, tfs,
                                                 substrate_radius,
                                                 env_conditions[env_conditions_id],
                                                 len(env_conditions), plastic_body, plastic_brain,
                                                 knockout)

                            distance = self.measure_distance(original_substrate, knockout_substrate)

                            await analyze(knockout_phenotype, experiment_name, run, gen, r.DbEAOptimizerIndividual.individual_id,
                                          knockstring, geno_size, len(promotors), ranking, distance)

    def _control(self, dt: float, control: ActorControl) -> None:
        self._controller.step(dt)
        control.set_dof_targets(0, 0, self._controller.get_dof_targets())

    def measure_distance(self, original_substrate, knockout_substrate):

        keys_first = set(original_substrate.keys())
        keys_second = set(knockout_substrate.keys())
        intersection = keys_first & keys_second
        disjunct_first = [a for a in keys_first if a not in intersection]
        disjunct_second = [b for b in keys_second if b not in intersection]
        body_changes = len(disjunct_first) + len(disjunct_second)

        for i in intersection:
            if hasattr(original_substrate[i], '_absolute_rotation'):
                if type(original_substrate[i]) != type(knockout_substrate[i]) or \
                    original_substrate[i]._absolute_rotation != knockout_substrate[i]._absolute_rotation :
                    body_changes += 1

        return body_changes


async def main() -> None:

    sim = Simulator()
    await sim.simulate()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



