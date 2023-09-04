from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration,\
    DbEAOptimizerIndividual, DbEAOptimizer, DbEnvconditions, DbEAOptimizerParent
from revolve2.core.modular_robot.render.render import Render
from genotype import GenotypeSerializer, develop
from revolve2.core.database.serializers import DbFloat

import os
import sys
import argparse
from ast import literal_eval
import math

async def main(parser) -> None:

    args = parser.parse_args()

    study = args.study
    experiments_name = args.experiments.split(',')
    runs = list(map(int, args.runs.split(',')))
    generations = [list(map(int, args.generations.split(',')))[-1]]
    mainpath = args.mainpath
    genealogy_length = 3

    # TODO: works only for asexual reproduction
    async def genealogy() -> None:

        for idsy, experiment_name in enumerate(experiments_name):
            print(experiment_name)
            for run in runs:
                print(' run: ', run)

                path = f'{mainpath}/{study}/analysis/genealogy/{experiment_name}/run_{run}'
                if not os.path.exists(path):
                    os.makedirs(path)

                db = open_async_database_sqlite(f'{mainpath}/{study}/{experiment_name}/run_{run}')

                for gen in generations:
                    print('  gen: ', gen)
                    path_gen = f'{path}/gen_{gen}'
                    if not os.path.exists(path_gen):
                        os.makedirs(path_gen)

                    async with AsyncSession(db) as session:

                        rows = ((await session.execute(select(DbEnvconditions).order_by(DbEnvconditions.id))).all())
                        env_conditions = {}
                        for c_row in rows:
                            env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)
                            path_env = f'{path}/gen_{gen}/env{c_row[0].id}'
                            if not os.path.exists(path_env):
                                os.makedirs(path_env)

                        rows = (
                            (await session.execute(select(DbEAOptimizer))).all()
                        )
                        max_modules = rows[0].DbEAOptimizer.max_modules
                        substrate_radius = rows[0].DbEAOptimizer.substrate_radius
                        plastic_body = rows[0].DbEAOptimizer.plastic_body
                        plastic_brain = rows[0].DbEAOptimizer.plastic_brain

                        query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizerParent)\
                            .filter(DbEAOptimizerGeneration.generation_index.in_([gen]) \
                                   & (DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                   & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                                    & (DbEAOptimizerIndividual.individual_id == DbEAOptimizerParent.child_individual_id)
                                   )
                        query = query.order_by(DbEAOptimizerIndividual.individual_id.asc())

                        leafs = ((await session.execute(query)).all())

                        for leaf in leafs:

                            leaf_id = leaf.DbEAOptimizerParent.child_individual_id
                            current_child = leaf_id

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
                                                                   plastic_brain, )
                            render = Render()
                            img_path = f'{path_gen}/env{leaf.DbEAOptimizerGeneration.env_conditions_id}/' \
                                       f'{leaf_id}_0_{current_child}.png'
                            render.render_robot(phenotype.body.core, img_path)

                            for idxb in range(1, genealogy_length + 1):

                                query = select(DbEAOptimizerParent, DbEAOptimizerIndividual). \
                                    filter((DbEAOptimizerParent.child_individual_id == current_child)
                                           & (DbEAOptimizerIndividual.individual_id == DbEAOptimizerParent.parent_individual_id)
                                           )

                                parent = ((await session.execute(query)).all())

                                if len(parent) > 0:
                                    parent_id = parent[0].DbEAOptimizerParent.parent_individual_id
                                    print(leaf_id, idxb, parent_id)

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
                                                                           plastic_brain, )
                                    render = Render()
                                    img_path = f'{path_gen}/env{leaf.DbEAOptimizerGeneration.env_conditions_id}/' \
                                               f'{leaf_id}_{idxb}_{parent_id}.png'
                                    render.render_robot(phenotype.body.core, img_path)

                                    current_child = parent_id

    async def similarity() -> None:

        for idsy, experiment_name in enumerate(experiments_name):
            print(experiment_name)
            for run in runs:
                print(' run: ', run)

                path = f'{mainpath}/{study}/analysis/similarity/{experiment_name}/run_{run}'
                if not os.path.exists(path):
                    os.makedirs(path)

                db = open_async_database_sqlite(f'{mainpath}/{study}/{experiment_name}/run_{run}')

                for gen in generations:
                    print('  gen: ', gen)
                    path_gen = f'{path}/gen_{gen}'
                    if not os.path.exists(path_gen):
                        os.makedirs(path_gen)

                    async with AsyncSession(db) as session:

                        rows = ((await session.execute(select(DbEnvconditions).order_by(DbEnvconditions.id))).all())
                        env_conditions = {}
                        for c_row in rows:
                            env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)
                            path_env = f'{path}/gen_{gen}/env{c_row[0].id}'
                            if not os.path.exists(path_env):
                                os.makedirs(path_env)

                        rows = (
                            (await session.execute(select(DbEAOptimizer))).all()
                        )
                        max_modules = rows[0].DbEAOptimizer.max_modules
                        substrate_radius = rows[0].DbEAOptimizer.substrate_radius
                        plastic_body = rows[0].DbEAOptimizer.plastic_body
                        plastic_brain = rows[0].DbEAOptimizer.plastic_brain

                        query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizerParent, DbFloat)\
                            .filter(DbEAOptimizerGeneration.generation_index.in_([gen]) \
                                   & (DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                   & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                                    & (DbEAOptimizerIndividual.individual_id == DbEAOptimizerParent.child_individual_id)
                                    & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                   )
                        query = query.order_by(DbEAOptimizerIndividual.individual_id.asc())

                        leafs = ((await session.execute(query)).all())

                        diffs = []
                        for leaf in leafs:

                            leaf_id = leaf.DbEAOptimizerParent.child_individual_id

                            # parent
                            query = select(DbEAOptimizerParent, DbEAOptimizerIndividual, DbFloat). \
                                filter((DbEAOptimizerParent.child_individual_id == leaf_id)
                                       & (DbEAOptimizerIndividual.individual_id == DbEAOptimizerParent.parent_individual_id)
                                       & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                       )

                            parent = ((await session.execute(query)).all())

                            if len(parent) > 0:
                                parent_id = parent[0].DbEAOptimizerParent.parent_individual_id

                                diff = abs(leaf.DbFloat.symmetry - parent[0].DbFloat.symmetry) + \
                                       abs(leaf.DbFloat.proportion - parent[0].DbFloat.proportion) + \
                                       abs(leaf.DbFloat.coverage - parent[0].DbFloat.coverage) + \
                                       abs(leaf.DbFloat.extremities_prop - parent[0].DbFloat.extremities_prop) + \
                                       abs(leaf.DbFloat.hinge_prop - parent[0].DbFloat.hinge_prop) + \
                                       abs(leaf.DbFloat.hinge_ratio - parent[0].DbFloat.hinge_ratio) + \
                                       abs(leaf.DbFloat.branching_prop - parent[0].DbFloat.branching_prop) + \
                                       abs(leaf.DbFloat.extensiveness_prop - parent[0].DbFloat.extensiveness_prop)

                                if diff > 0:
                                    diffs.append(diff)
                                    print(leaf_id, parent_id, diff)

                                    genotype = (
                                        await GenotypeSerializer.from_database(
                                            session, [leaf_id]
                                        )
                                    )[0]
                                    phenotype, queried_substrate = develop(genotype, genotype.mapping_seed,
                                                                           max_modules,
                                                                           substrate_radius,
                                                                           env_conditions[
                                                                               leaf.DbEAOptimizerGeneration.env_conditions_id],
                                                                           len(env_conditions), plastic_body,
                                                                           plastic_brain, )
                                    render = Render()
                                    img_path = f'{path_gen}/env{leaf.DbEAOptimizerGeneration.env_conditions_id}/' \
                                               f'{leaf_id}_0_{leaf_id}.png'
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
                                                                           plastic_brain, )
                                    render = Render()
                                    img_path = f'{path_gen}/env{leaf.DbEAOptimizerGeneration.env_conditions_id}/' \
                                               f'{leaf_id}_1_{parent_id}.png'
                                    render.render_robot(phenotype.body.core, img_path)

                        if len(diffs) > 0:
                            print('mean diff', sum(diffs) / len(diffs), 'total', len(diffs))
                        else:
                            print('mean diff', 0)

    #await genealogy()
    await similarity()

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