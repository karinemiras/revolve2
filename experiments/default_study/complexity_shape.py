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
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
import os
from ast import literal_eval
import math

from revolve2.runners.isaacgym import LocalRunner as LocalRunnerI


from body_spider import *


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
        self.generations = [100]
        test_robots = []
        self.mainpath = args.mainpath

        self.bests = 1
        # 'all' selects best from all individuals
        # 'gens' selects best from chosen generations
        self.bests_type = 'gens'
        self.ranking = ['best', 'worst']

        self.path = f'{self.mainpath}/{self.study}/analysis/comp/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.pfile = f'{self.path}/comp.csv'
        header = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id', 'complexity']
        with open(self.pfile, 'w') as file:
            file.write(','.join(map(str, header)))
            file.write('\n')

        for ids, experiment_name in enumerate(self.experiments_name):
            print('\n', experiment_name)
            for run in self.runs:
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

                for ranking in self.ranking:

                    query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat) \
                        .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                                & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                & DbEAOptimizerGeneration.generation_index.in_([gen])
                                )

                    if len(test_robots) > 0:
                        query = query.filter(DbEAOptimizerIndividual.individual_id.in_(test_robots))

                    print(' \n', ranking)
                    if ranking == 'best':
                        # if seasonal setup, criteria is seasonal pareto
                        if len(rows) > 1:
                            query = query.order_by(
                                                   # CAN ALSO USE SOME OTHER CRITERIA INSTEAD OF SEASONAL
                                                   DbEAOptimizerGeneration.seasonal_dominated.desc(),
                                                   DbEAOptimizerGeneration.individual_id.asc(),
                                                   DbEAOptimizerGeneration.env_conditions_id.asc())
                        else:
                            query = query.order_by(DbFloat.disp_y.desc())
                    else:
                        if len(rows) > 1:
                            query = query.order_by(
                                                   DbEAOptimizerGeneration.seasonal_dominated.asc(),
                                                   DbEAOptimizerGeneration.individual_id.asc(),
                                                   DbEAOptimizerGeneration.env_conditions_id.asc())
                        else:
                            query = query.order_by(DbFloat.disp_y.asc())

                    rows = ((await session.execute(query)).all())

                    num_lines = self.bests * len(env_conditions)
                    for idx, r in enumerate(rows[0:num_lines]):

                        env_conditions_id = r.DbEAOptimizerGeneration.env_conditions_id
                        print(f'\n  rk:{idx+1} ' \
                                 f' id:{r.DbEAOptimizerIndividual.individual_id} ' \
                                 f' birth:{r.DbFloat.birth} ' \
                                 f' gen:{r.DbEAOptimizerGeneration.generation_index} ' \
                                 f' cond:{env_conditions_id} ' \
                                 f' dom:{r.DbEAOptimizerGeneration.seasonal_dominated} ' \
                                 f' speed_y:{r.DbFloat.speed_y} ' \
                                 f' disp_y:{r.DbFloat.disp_y} ' \
                              )

                        genotype = (
                            await GenotypeSerializer.from_database(
                                session, [r.DbEAOptimizerIndividual.genotype_id]
                            )
                        )[0]

                        original_phenotype, original_substrate, genes = \
                            develop_knockout(genotype, genotype.mapping_seed, max_modules, tfs,
                                             substrate_radius, env_conditions[env_conditions_id],
                                             len(env_conditions), plastic_body, plastic_brain,
                                             None)
                    print(list(original_substrate.keys()))
                    coordinates = list(original_substrate.keys())

                    # Define the squares
                    def create_square(x, y):
                        return Polygon([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)])

                    # Create square polygons
                    polygons = [create_square(x, y) for x, y in coordinates]

                    # Combine all polygons into one shape
                    shape = unary_union(polygons)

                    # Extract the exterior coordinates of the shape
                    if shape.geom_type == 'Polygon':
                        exterior_coords = np.array(shape.exterior.coords)
                    elif shape.geom_type == 'MultiPolygon':
                        exterior_coords = np.concatenate([np.array(p.exterior.coords) for p in shape.geoms])
                    else:
                        raise ValueError("Shape must be a Polygon or MultiPolygon")

                    # Function to remove collinear points and find unique boundary points
                    def remove_collinear_points(coords):
                        unique_points = []
                        for i in range(len(coords)):
                            p1 = coords[i]
                            p2 = coords[(i + 1) % len(coords)]
                            p3 = coords[(i + 2) % len(coords)]

                            # Vector calculations
                            v1 = np.array(p2) - np.array(p1)
                            v2 = np.array(p3) - np.array(p2)

                            # Check for collinearity
                            if not np.allclose(np.cross(v1, v2), 0):
                                unique_points.append(p2)

                        return np.array(unique_points)

                    # Function to count and draw 90-degree changes
                    def count_and_draw_90_degree_changes(coords):
                        changes = 0
                        marked_segments = []
                        for i in range(len(coords)):
                            p1 = coords[i]
                            p2 = coords[(i + 1) % len(coords)]
                            p3 = coords[(i + 2) % len(coords)]

                            v1 = np.array(p2) - np.array(p1)
                            v2 = np.array(p3) - np.array(p2)

                            # Compute dot product
                            dot_product = np.dot(v1, v2)

                            if np.isclose(dot_product, 0):
                                changes += 1
                                marked_segments.append((p1, p2))
                                marked_segments.append((p2, p3))

                        return changes, marked_segments

                    # Remove collinear points
                    unique_coords = remove_collinear_points(exterior_coords)

                    # Calculate the number of 90-degree changes and marked segments
                    num_changes, marked_segments = count_and_draw_90_degree_changes(unique_coords)

                    # Plot the shape with direction changes highlighted
                    plt.figure()
                    # Draw the full boundary of the shape
                    plt.plot(exterior_coords[:, 0], exterior_coords[:, 1], 'b-')
                    # Draw the segments with 90-degree changes in black
                    for (start, end) in marked_segments:
                        plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)

                    plt.xlabel('X-axis')
                    plt.ylabel('Y-axis')
                    plt.title('Envelope of the Shape with Direction Changes')
                    plt.legend()
                    plt.grid(True)
                    plt.gca().set_aspect('equal')
                    plt.savefig(f'{self.path}/{experiment_name}_{run}_{gen}_{ranking}_{r.DbEAOptimizerIndividual.individual_id}.png', format='png')

                    print(f"Number of 90-degree changes: {num_changes}")
                    with open(f'{self.path}/comp.csv', 'a') as f:
                        f.write(
                            f"{experiment_name},{run},{gen},{ranking},{r.DbEAOptimizerIndividual.individual_id},{num_changes} \n")


async def main() -> None:

    sim = Simulator()
    await sim.simulate()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



