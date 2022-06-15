from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual
from revolve2.core.modular_robot.render.render import Render
#TODO: make import based on param and move file to anal_resources
from opt_mod_1.genotype import GenotypeSerializer, develop
from revolve2.core.config import Config

import os

async def main() -> None:

    args = Config()._get_params()

    study = 'default_study'
    experiments_name = ['default_experiment']
    runs = [1]
    generations = [2, 3]

    for experiment_name in experiments_name:
        for run in runs:

            path = f'data/{study}/analysis/snapshots/{experiment_name}/run_{run}'
            if not os.path.exists(path):
                os.makedirs(path)

            db = open_async_database_sqlite(f'./data/{study}/{experiment_name}/run_{run}')

            for gen in generations:
                path_gen = f'{path}/gen_{gen}'
                if os.path.exists(path_gen):
                    print(f'{path_gen} already exists!')
                else:
                    os.makedirs(path_gen)

                    async with AsyncSession(db) as session:
                        rows = (
                            (await session.execute(select(DbEAOptimizerGeneration, DbEAOptimizerIndividual)
                                                   .filter(DbEAOptimizerGeneration.generation_index.in_([gen]))
                                                   .filter(DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                                   .order_by(DbEAOptimizerGeneration.pool_dominated_individuals.desc())


                            )).all()
                        )

                        for idx, r in enumerate(rows):
                            genotype = (
                                await GenotypeSerializer.from_database(
                                    session, [r.DbEAOptimizerIndividual.genotype_id]
                                )
                            )[0]

                            phenotype = develop(genotype, args.max_modules)
                            render = Render()
                            img_path = f'{path_gen}/{idx}_{r.DbEAOptimizerIndividual.individual_id}.png'
                            render.render_robot(phenotype.body.core, img_path)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
