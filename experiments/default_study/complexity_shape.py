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
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib.patches import Polygon as mpl_polygon
import os
from ast import literal_eval
import math

from revolve2.runners.isaacgym import LocalRunner as LocalRunnerI


from body_spider import *


# TODO: make this part of morphological measures .py later.
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
        self.generations = list(range(0, 101))
        test_robots = []
        self.mainpath = args.mainpath

        self.bests = 100
        # 'all' selects best from all individuals
        # 'gens' selects best from chosen generations
        self.bests_type = 'gens'

        self.path = f'{self.mainpath}/{self.study}/analysis/complexity/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.pfile = f'{self.path}/complexity.csv'
        header = ['experiment_name', 'run', 'gen', 'individual_id', 'disp_y', 'geno_size',
                  'complexity_env', 'complexity_branch', 'symmetry', 'extremities_prop']

        with open(self.pfile, 'w') as file:
            file.write(','.join(map(str, header)))
            file.write('\n')

        for ids, experiment_name in enumerate(self.experiments_name):
            print('\n', experiment_name)
            for run in self.runs[ids]:
                print('\nrun: ', run)

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
                            )

                if len(test_robots) > 0:
                    query = query.filter(DbEAOptimizerIndividual.individual_id.in_(test_robots))

                # if seasonal setup, criteria is seasonal pareto
                if len(rows) > 1:
                    query = query.order_by(
                                           # CAN ALSO USE SOME OTHER CRITERIA INSTEAD OF SEASONAL
                                           DbEAOptimizerGeneration.seasonal_dominated.desc(),
                                           DbEAOptimizerGeneration.individual_id.asc(),
                                           DbEAOptimizerGeneration.env_conditions_id.asc())
                else:
                    query = query.order_by(DbFloat.disp_y.desc())

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
                    #          f' disp_y:{r.DbFloat.disp_y} ' \
                    #       )

                    genotype = (
                        await GenotypeSerializer.from_database(
                            session, [r.DbEAOptimizerIndividual.genotype_id]
                        )
                    )[0]
                    geno_size = len(genotype.body.genotype)

                    original_phenotype, original_substrate = \
                        develop(genotype, genotype.mapping_seed, max_modules, tfs,
                                         substrate_radius, env_conditions[env_conditions_id],
                                         len(env_conditions), plastic_body, plastic_brain)

                    all_children = []

                    def finalize_recur(module, all_children):
                        children = sum([1 for i in module.children if i is not None])
                        if children > 0:
                            all_children.append(children)
                        for i, child in enumerate(module.children):
                            if child is not None:
                                finalize_recur(child, all_children)

                    module = original_phenotype.body.core
                    finalize_recur(module, all_children)
                    if len(all_children) > 0:
                        avg_children = sum(all_children) / len(all_children)

                  #  print(list(original_substrate.keys()))
                    coordinates = list(original_substrate.keys())

                    def plot_and_save_boundary(coordinates):

                        # Create squares centered at each coordinate
                        squares = [
                            Polygon([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)])
                            for x, y in
                            coordinates]
                        # Compute the union of all squares
                        union = unary_union(squares)

                        # Create a matplotlib patch for the boundary
                        if isinstance(union, Polygon):
                            boundary_coords = np.array(union.exterior.coords)
                        elif isinstance(union, MultiPolygon):
                            # Take the exterior coordinates of each polygon and merge them
                            boundary_coords = np.vstack([np.array(p.exterior.coords) for p in union])
                        else:
                            raise ValueError("Union result is neither a Polygon nor a MultiPolygon")

                        # Calculate number of 90-degree turns
                        def count_90_degree_turns(coords):
                            directions = np.diff(coords, axis=0)
                            angles = np.arctan2(directions[:, 1], directions[:, 0])
                            angle_diffs = np.diff(np.concatenate((angles, [angles[0]])))
                            angle_diffs = np.abs((angle_diffs + np.pi) % (2 * np.pi) - np.pi)
                            num_turns = np.sum(np.isclose(angle_diffs, np.pi / 2))
                            return num_turns

                        # Calculate and print the number of 90-degree turns
                        num_turns = count_90_degree_turns(boundary_coords)
                        #print(f"Number of 90-degree turns: {num_turns}")

                        if gen == 100 and idx == 0:

                            fig, ax = plt.subplots()

                            # Plot the squares with grey lines
                            for square in squares:
                                x, y = square.exterior.xy
                                ax.plot(x, y, 'black', lw=1)

                            # Plot the boundary with a thicker red line
                            boundary_polygon = mpl_polygon(boundary_coords, edgecolor='red', facecolor='none',
                                                           linewidth=4)
                            ax.add_patch(boundary_polygon)

                            # Set limits and aspect
                            x_min, y_min = np.min(boundary_coords, axis=0) - 1
                            x_max, y_max = np.max(boundary_coords, axis=0) + 1
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(y_min, y_max)
                            ax.set_aspect('equal')

                            ax.grid(False)#, color='lightgrey', linewidth=0.5)

                            plt.savefig(
                                f'{self.path}/{experiment_name}_{run}_{gen}_{r.DbEAOptimizerIndividual.individual_id}_{num_turns}.png',
                                format='png')
                            plt.close()

                        return num_turns

                    num_turns = plot_and_save_boundary(coordinates)

                    with open(f'{self.path}/complexity.csv', 'a') as f:
                        f.write(
                            f"{experiment_name},{run},{gen},"
                            f"{r.DbEAOptimizerIndividual.individual_id},{r.DbFloat.disp_y},{geno_size},{num_turns},{avg_children},{r.DbFloat.symmetry},{r.DbFloat.extremities_prop}\n")


async def main() -> None:

    sim = Simulator()
    await sim.simulate()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



