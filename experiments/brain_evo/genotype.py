from dataclasses import dataclass
from random import Random
from typing import List

import multineat
import sqlalchemy
import math
from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.modular_robot import ModularRobot, ActiveHinge, Body, Brick
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

from revolve2.core.database import IncompatibleError, Serializer
from revolve2.core.modular_robot import ModularRobot
from revolve2.genotypes.cppnwin import Genotype as CppnwinGenotype
from revolve2.genotypes.cppnwin import GenotypeSerializer as CppnwinGenotypeSerializer
from revolve2.genotypes.cppnwin import crossover_v1, mutate_v1
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v2 import Develop as body_develop
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v2 import (
    random_v1 as body_random,
)
from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_ann import (
    develop_v1 as brain_develop,
)
from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_ann import (
    random_v1 as brain_random,
)


def _make_multineat_params() -> multineat.Parameters:
    multineat_params = multineat.Parameters()

    multineat_params.OverallMutationRate = 1
    multineat_params.MutateAddLinkProb = 0.5
    multineat_params.MutateRemLinkProb = 0.5
    multineat_params.MutateAddNeuronProb = 0.2
    multineat_params.MutateRemSimpleNeuronProb = 0.2
    multineat_params.RecurrentProb = 0.0
    multineat_params.MutateWeightsProb = 0.8
    multineat_params.WeightMutationMaxPower = 0.5
    multineat_params.WeightReplacementMaxPower = 1.0
    multineat_params.MutateActivationAProb = 0#0.5
    multineat_params.ActivationAMutationMaxPower = 0.5
    multineat_params.MinActivationA = 0.05
    multineat_params.MaxActivationA = 6.0
    multineat_params.MaxWeight = 8.0

    multineat_params.MutateNeuronActivationTypeProb = 0#0.03

    multineat_params.MutateOutputActivationFunction = False

    # multineat_params.ActivationFunction_SignedSigmoid_Prob = 1.0
    # multineat_params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    # multineat_params.ActivationFunction_Tanh_Prob = 1.0
    # multineat_params.ActivationFunction_TanhCubic_Prob = 0.0
    # multineat_params.ActivationFunction_SignedStep_Prob = 1.0
    # multineat_params.ActivationFunction_UnsignedStep_Prob = 0.0
    # multineat_params.ActivationFunction_SignedGauss_Prob = 1.0
    # multineat_params.ActivationFunction_UnsignedGauss_Prob = 0.0
    # multineat_params.ActivationFunction_Abs_Prob = 1.0
    # multineat_params.ActivationFunction_SignedSine_Prob = 1.0
    # multineat_params.ActivationFunction_UnsignedSine_Prob = 0.0
    # multineat_params.ActivationFunction_Linear_Prob = 1.0

    multineat_params.MutateNeuronTraitsProb = 0.0
    multineat_params.MutateLinkTraitsProb = 0.0

    multineat_params.AllowLoops = False

    return multineat_params


_MULTINEAT_PARAMS = _make_multineat_params()


@dataclass
class Genotype:
    body: CppnwinGenotype
    brain: CppnwinGenotype
    mapping_seed: int


class GenotypeSerializer(Serializer[Genotype]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await CppnwinGenotypeSerializer.create_tables(session)

    @classmethod
    def identifying_table(cls) -> str:
        return DbGenotype.__tablename__

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[Genotype]
    ) -> List[int]:
        body_ids = await CppnwinGenotypeSerializer.to_database(
            session, [o.body for o in objects]
        )
        brain_ids = await CppnwinGenotypeSerializer.to_database(
            session, [o.brain for o in objects]
        )
        mapping_seeds = [o.mapping_seed for o in objects]

        dbgenotypes = [
            DbGenotype(body_id=body_id, brain_id=brain_id, mapping_seed=mapping_seed)
            for body_id, brain_id, mapping_seed in zip(body_ids, brain_ids, mapping_seeds)
        ]

        session.add_all(dbgenotypes)
        await session.flush()
        ids = [
            dbfitness.id for dbfitness in dbgenotypes if dbfitness.id is not None
        ]  # cannot be none because not nullable. check if only there to silence mypy.
        assert len(ids) == len(objects)  # but check just to be sure
        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[Genotype]:
        rows = (
            (await session.execute(select(DbGenotype).filter(DbGenotype.id.in_(ids))))
            .scalars()
            .all()
        )

        if len(rows) != len(ids):
            raise IncompatibleError()

        id_map = {t.id: t for t in rows}
        body_ids = [id_map[id].body_id for id in ids]
        brain_ids = [id_map[id].brain_id for id in ids]
        mapping_seeds = [t.mapping_seed for t in rows]

        body_genotypes = await CppnwinGenotypeSerializer.from_database(
            session, body_ids
        )
        brain_genotypes = await CppnwinGenotypeSerializer.from_database(
            session, brain_ids
        )

        genotypes = [
            Genotype(body, brain, mapping_seed)
            for body, brain, mapping_seed in zip(body_genotypes, brain_genotypes, mapping_seeds)
        ]

        return genotypes


def random(
    innov_db_body: multineat.InnovationDatabase,
    innov_db_brain: multineat.InnovationDatabase,
    rng: Random,
    num_initial_mutations: int,
    n_env_conditions: int,
    plastic_body: int,
    plastic_brain: int,
) -> Genotype:
    multineat_rng = _multineat_rng_from_random(rng)

    body = body_random(
        innov_db_body,
        multineat_rng,
        _MULTINEAT_PARAMS,
        multineat.ActivationFunction.TANH,
        num_initial_mutations,
        n_env_conditions,
        plastic_body,
    )

    brain = brain_random(
        innov_db_brain,
        multineat_rng,
        _MULTINEAT_PARAMS,
        #multineat.ActivationFunction.SIGNED_SINE,
        multineat.ActivationFunction.TANH,
        num_initial_mutations,
        n_env_conditions,
        plastic_brain,
    )

    mapping_seed = rng.randint(0, 2 ** 31)

    return Genotype(body, brain, mapping_seed)


def mutate(
    genotype: Genotype,
    innov_db_body: multineat.InnovationDatabase,
    innov_db_brain: multineat.InnovationDatabase,
    rng: Random,
) -> Genotype:
    multineat_rng = _multineat_rng_from_random(rng)

    return Genotype(
        mutate_v1(genotype.body, _MULTINEAT_PARAMS, innov_db_body, multineat_rng),
        mutate_v1(genotype.brain, _MULTINEAT_PARAMS, innov_db_brain, multineat_rng),
        genotype.mapping_seed,
    )


def crossover(
    parent1: Genotype,
    parent2: Genotype,
    rng: Random,
) -> Genotype:
    multineat_rng = _multineat_rng_from_random(rng)

    return Genotype(
        crossover_v1(
            parent1.body,
            parent2.body,
            _MULTINEAT_PARAMS,
            multineat_rng,
            False,
            False,
        ),
        crossover_v1(
            parent1.brain,
            parent2.brain,
            _MULTINEAT_PARAMS,
            multineat_rng,
            False,
            False,
        ),
        parent1.mapping_seed,
    )


def develop(genotype: Genotype, querying_seed: int, max_modules: int, substrate_radius: str, env_condition: list,
            n_env_conditions: int, plastic_body: int, plastic_brain: int) -> ModularRobot:

    #using a fixed body for every robot (thus ignoring body genotype)

    body = Body()
    body.core._id = 1

    ###
    body.core.front = ActiveHinge(math.pi / 2.0)
    body.core.front._id = 2
    body.core.front._absolute_rotation = 90

    body.core.front.attachment = Brick(0.0)
    body.core.front.attachment._id = 3
    body.core.front.attachment._absolute_rotation = 0

    body.core.front.attachment.front = ActiveHinge(math.pi / 2.0)
    body.core.front.attachment.front._id = 4
    body.core.front.attachment.front._absolute_rotation = 0

    body.core.front.attachment.front.attachment = Brick(0.0)
    body.core.front.attachment.front.attachment._id = 5
    body.core.front.attachment.front.attachment._absolute_rotation = 0

    ###
    body.core.right = ActiveHinge(math.pi / 2.0)
    body.core.right._id = 6
    body.core.right._absolute_rotation = 90

    body.core.right.attachment = Brick(0.0)
    body.core.right.attachment._id = 7
    body.core.right.attachment._absolute_rotation = 0

    body.core.right.attachment.front = ActiveHinge(math.pi / 2.0)
    body.core.right.attachment.front._id = 8
    body.core.right.attachment.front._absolute_rotation = 0

    body.core.right.attachment.front.attachment = Brick(0.0)
    body.core.right.attachment.front.attachment._id = 9
    body.core.right.attachment.front.attachment._absolute_rotation = 0

    ###
    body.core.back = ActiveHinge(math.pi / 2.0)
    body.core.back._id = 10
    body.core.back._absolute_rotation = 90

    body.core.back.attachment = Brick(0.0)
    body.core.back.attachment._id = 11
    body.core.back.attachment._absolute_rotation = 0

    body.core.back.attachment.front = ActiveHinge(math.pi / 2.0)
    body.core.back.attachment.front._id = 12
    body.core.back.attachment.front._absolute_rotation = 0

    body.core.back.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment._id = 13
    body.core.back.attachment.front.attachment._absolute_rotation = 0

    ###
    body.core.left = ActiveHinge(math.pi / 2.0)
    body.core.left._id = 14
    body.core.left._absolute_rotation = 90

    body.core.left.attachment = Brick(0.0)
    body.core.left.attachment._id = 15
    body.core.left.attachment._absolute_rotation = 0

    body.core.left.attachment.front = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment.front._id = 16
    body.core.left.attachment.front._absolute_rotation = 0

    body.core.left.attachment.front.attachment = Brick(0.0)
    body.core.left.attachment.front.attachment._id = 17
    body.core.left.attachment.front.attachment._absolute_rotation = 0

    body.finalize()
    brain = brain_develop(genotype.brain, body, env_condition, n_env_conditions, plastic_brain)

    return ModularRobot(body, brain), []


def _multineat_rng_from_random(rng: Random) -> multineat.RNG:
    multineat_rng = multineat.RNG()
    multineat_rng.Seed(rng.randint(0, 2**31))
    return multineat_rng


DbBase = declarative_base()


class DbGenotype(DbBase):
    __tablename__ = "genotype"

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )

    body_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    brain_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    mapping_seed = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
