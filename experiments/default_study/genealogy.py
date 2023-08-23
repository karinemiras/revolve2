from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer, DbEnvconditions, DbEAOptimizerParent
from revolve2.core.modular_robot.render.render import Render
from genotype import GenotypeSerializer, develop
from revolve2.core.database.serializers import DbFloat

import os
import sys
import argparse
from ast import literal_eval


async def main(parser) -> None:

    args = parser.parse_args()

    study = args.study
    experiments_name = args.experiments.split(',')
    runs = list(map(int, args.runs.split(',')))
    generations = [list(map(int, args.generations.split(',')))[-1]]
    mainpath = args.mainpath
    max_best = 10
    genealogy_length = 10

    for idsy, experiment_name in enumerate(experiments_name):
        print(experiment_name)
        for run in runs:
            print(' run: ', run)

            path = f'{mainpath}/{study}/analysis/genealogy/{experiment_name}/' \
                   f'run_{run}'
            if not os.path.exists(path):
                os.makedirs(path)

            db = open_async_database_sqlite(f'{mainpath}/{study}/{experiment_name}/run_{run}')

            for gen in generations:
                print('  gen: ', gen)
                path_gen = f'{path}/gen_{gen}'
                if os.path.exists(path_gen):
                    print(f'{path_gen} already exists!')
                else:
                    os.makedirs(path_gen)

                    async with AsyncSession(db) as session:

                        rows = ((await session.execute(select(DbEnvconditions).order_by(DbEnvconditions.id))).all())
                        env_conditions = {}
                        for c_row in rows:
                            env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)
                            os.makedirs(f'{path}/gen_{gen}/env{c_row[0].id}')

                        rows = (
                            (await session.execute(select(DbEAOptimizer))).all()
                        )
                        max_modules = rows[0].DbEAOptimizer.max_modules
                        substrate_radius = rows[0].DbEAOptimizer.substrate_radius
                        plastic_body = rows[0].DbEAOptimizer.plastic_body
                        plastic_brain = rows[0].DbEAOptimizer.plastic_brain

                        query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat, DbEAOptimizerParent)\
                            .filter(DbEAOptimizerGeneration.generation_index.in_([gen]) \
                                   & (DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                   & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                                   & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                   & (DbEAOptimizerIndividual.individual_id == DbEAOptimizerParent.child_individual_id)
                                   )

                        if len(env_conditions) > 1:
                            query = query.order_by(DbEAOptimizerGeneration.seasonal_dominated.desc(),
                                                   DbEAOptimizerGeneration.individual_id.asc(),
                                                   DbEAOptimizerGeneration.env_conditions_id.asc())
                        else:
                            query = query.order_by(DbFloat.speed_y.desc())

                        leafs = ((await session.execute(query)).all())

                        diffs = []
                        #for idx, leaf in enumerate(leafs[0:max_best]):
                        for idx, leaf in enumerate(leafs):
                            leaf_id = leaf.DbEAOptimizerParent.child_individual_id
                            current_child = leaf_id

                            for idxb in range(0, genealogy_length+1):

                                query = select(DbEAOptimizerParent, DbEAOptimizerIndividual, DbFloat). \
                                    filter((DbEAOptimizerParent.child_individual_id == current_child)
                                           & (DbEAOptimizerIndividual.individual_id == DbEAOptimizerParent.parent_individual_id)
                                           & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                           )

                                parent = ((await session.execute(query)).all())

                                if len(parent) > 0:
                                    parent_id = parent[0].DbEAOptimizerParent.parent_individual_id
                                    diff = abs(leaf.DbFloat.symmetry - parent[0].DbFloat.symmetry)+ \
                                           abs(leaf.DbFloat.proportion - parent[0].DbFloat.proportion) + \
                                           abs(leaf.DbFloat.coverage - parent[0].DbFloat.coverage) + \
                                           abs(leaf.DbFloat.extremities_prop - parent[0].DbFloat.extremities_prop) + \
                                           abs(leaf.DbFloat.hinge_prop - parent[0].DbFloat.hinge_prop) + \
                                           abs(leaf.DbFloat.hinge_ratio - parent[0].DbFloat.hinge_ratio) + \
                                           abs(leaf.DbFloat.branching_prop - parent[0].DbFloat.branching_prop) + \
                                           abs(leaf.DbFloat.extensiveness_prop - parent[0].DbFloat.extensiveness_prop)

                                    if diff > 0:
                                        diffs.append(diff)

                                        genotype = (
                                            await GenotypeSerializer.from_database(
                                                session, [current_child]
                                            )
                                        )[0]
                                        phenotype, queried_substrate = develop(genotype, genotype.mapping_seed,
                                                                               max_modules,
                                                                               substrate_radius,
                                                                               env_conditions[
                                                                                   leaf.DbEAOptimizerGeneration.env_conditions_id],
                                                                               len(env_conditions), plastic_body,
                                                                               plastic_brain,  )
                                        render = Render()
                                        img_path = f'{path_gen}/env{leaf.DbEAOptimizerGeneration.env_conditions_id}/' \
                                                   f'{leaf_id}_{idxb}_{current_child}.png'
                                        render.render_robot(phenotype.body.core, img_path)

                                        genotype = (
                                            await GenotypeSerializer.from_database(
                                                session, [parent_id]
                                            )
                                        )[0]
                                        phenotype, queried_substrate = develop(genotype, genotype.mapping_seed,
                                                                               max_modules,
                                                                               substrate_radius,
                                                                               env_conditions[
                                                                                   leaf.DbEAOptimizerGeneration.env_conditions_id],
                                                                               len(env_conditions), plastic_body,
                                                                               plastic_brain,  )
                                        render = Render()
                                        img_path = f'{path_gen}/env{leaf.DbEAOptimizerGeneration.env_conditions_id}/' \
                                                   f'{leaf_id}_{idxb}_{parent_id}.png'
                                        render.render_robot(phenotype.body.core, img_path)

                                    current_child = parent_id
                                else:
                                    current_child = -1
                        if len(diffs) > 0:
                            print('mean', sum(diffs)/len(diffs))
                        else:
                            print('mean', 0)



if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("study")
    parser.add_argument("experiments")
    parser.add_argument("runs")
    parser.add_argument("generations")
    parser.add_argument("mainpath")

    asyncio.run(main(parser))

# can be run from root