from dataclasses import dataclass
from random import Random
from typing import List

import multineat
import sqlalchemy
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

from revolve2.core.database import IncompatibleError, Serializer
from revolve2.core.modular_robot import ModularRobot
from revolve2.genotypes.cppnwin import Genotype as CppnwinGenotype
from revolve2.genotypes.cppnwin import GenotypeSerializer as CppnwinGenotypeSerializer

from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v2 import Develop as cppn_body_develop
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v2 import (random_v1 as cppn_body_random)
from revolve2.genotypes.cppnwin import crossover_v1, mutate_v1

from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_cpg_v1 import (
    develop_v1 as brain_cpg_develop,
)
from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_cpg_v1 import (
    random_v1 as brain_cpg_random,
)

from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_ann import (
    develop_v1 as brain_ann_develop,
)
from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_ann import (
    random_v1 as brain_ann_random,
)

from revolve2.genotypes.cppnwin.modular_robot.genotype_grn import Develop as grn_develop
from revolve2.genotypes.cppnwin.modular_robot.genotype_grn import (random_v1 as grn_random)
from revolve2.genotypes.cppnwin import mutate_grn


from body_spider import *


def _make_multineat_params_cppn() -> multineat.Parameters:
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
    multineat_params.MutateActivationAProb = 0.5
    multineat_params.ActivationAMutationMaxPower = 0.5
    multineat_params.MinActivationA = 0.05
    multineat_params.MaxActivationA = 6.0
    multineat_params.MaxWeight = 8.0

    multineat_params.MutateNeuronActivationTypeProb = 0.03

    multineat_params.MutateOutputActivationFunction = False

    multineat_params.ActivationFunction_SignedSigmoid_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    multineat_params.ActivationFunction_Tanh_Prob = 1.0
    multineat_params.ActivationFunction_TanhCubic_Prob = 0.0
    multineat_params.ActivationFunction_SignedStep_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedStep_Prob = 0.0
    multineat_params.ActivationFunction_SignedGauss_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedGauss_Prob = 0.0
    multineat_params.ActivationFunction_Abs_Prob = 1.0
    multineat_params.ActivationFunction_SignedSine_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedSine_Prob = 0.0
    multineat_params.ActivationFunction_Linear_Prob = 1.0

    multineat_params.MutateNeuronTraitsProb = 0.0
    multineat_params.MutateLinkTraitsProb = 0.0

    multineat_params.AllowLoops = False

    return multineat_params


def _make_multineat_params_ann() -> multineat.Parameters:
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
    multineat_params.MutateActivationAProb = 0
    multineat_params.ActivationAMutationMaxPower = 0.5
    multineat_params.MinActivationA = 0.05
    multineat_params.MaxActivationA = 6.0
    multineat_params.MaxWeight = 8.0
    multineat_params.MutateNeuronActivationTypeProb = 0
    multineat_params.MutateOutputActivationFunction = False
    multineat_params.MutateNeuronTraitsProb = 0.0
    multineat_params.MutateLinkTraitsProb = 0.0

    multineat_params.AllowLoops = False

    return multineat_params


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
    loop: str,
    body_phenotype: str,
) -> Genotype:
    multineat_rng = _multineat_rng_from_random(rng)

    if loop == 'open':
        _MULTINEAT_PARAMS_BRAIN = _make_multineat_params_cppn()

        brain = brain_cpg_random(
            innov_db_brain,
            multineat_rng,
            _MULTINEAT_PARAMS_BRAIN,
            multineat.ActivationFunction.SIGNED_SINE,
            num_initial_mutations,
            n_env_conditions,
            plastic_brain,
        )

    if loop == 'closed':
        _MULTINEAT_PARAMS_BRAIN = _make_multineat_params_ann()

        brain = brain_ann_random(
            innov_db_brain,
            multineat_rng,
            _MULTINEAT_PARAMS_BRAIN,
            multineat.ActivationFunction.TANH,
            num_initial_mutations,
            n_env_conditions,
            plastic_brain,
        )

    # TODO: when body is not evolvable, this gets unecessarily generated anyway

    # _MULTINEAT_PARAMS_BODY = _make_multineat_params_cppn()
    # body = cppn_body_random(
    #     innov_db_body,
    #     multineat_rng,
    #     _MULTINEAT_PARAMS_BODY,
    #     multineat.ActivationFunction.TANH,
    #     num_initial_mutations,
    #     n_env_conditions,
    #     plastic_body,
    # )

    body = grn_random(rng)

    mapping_seed = rng.randint(0, 2 ** 31)

    return Genotype(body, brain, mapping_seed)


def mutate(
    genotype: Genotype,
    innov_db_body: multineat.InnovationDatabase,
    innov_db_brain: multineat.InnovationDatabase,
    rng: Random,
    loop: str,
) -> Genotype:
    multineat_rng = _multineat_rng_from_random(rng)

    if loop == 'open':
        _MULTINEAT_PARAMS_BRAIN = _make_multineat_params_cppn()
    if loop == 'closed':
        _MULTINEAT_PARAMS_BRAIN = _make_multineat_params_ann()

    _MULTINEAT_PARAMS_BODY = _make_multineat_params_cppn()

    return Genotype(
        #mutate_v1(genotype.body, _MULTINEAT_PARAMS_BODY, innov_db_body, multineat_rng),
        mutate_grn(genotype.body, rng),
        mutate_v1(genotype.brain, _MULTINEAT_PARAMS_BRAIN, innov_db_brain, multineat_rng),
        genotype.mapping_seed,
    )


def crossover(
    parent1: Genotype,
    parent2: Genotype,
    rng: Random,
    loop: str,
) -> Genotype:
    multineat_rng = _multineat_rng_from_random(rng)

    if loop == 'open':
        _MULTINEAT_PARAMS_BRAIN = _make_multineat_params_cppn()
    if loop == 'closed':
        _MULTINEAT_PARAMS_BRAIN = _make_multineat_params_ann()

    _MULTINEAT_PARAMS_BODY = _make_multineat_params_cppn()

    return Genotype(
        crossover_v1(
            parent1.body,
            parent2.body,
            _MULTINEAT_PARAMS_BODY,
            multineat_rng,
            False,
            False,
        ),
        crossover_v1(
            parent1.brain,
            parent2.brain,
            _MULTINEAT_PARAMS_BRAIN,
            multineat_rng,
            False,
            False,
        ),
        parent1.mapping_seed,
    )


def develop(genotype: Genotype, querying_seed: int, max_modules: int, substrate_radius: str, env_condition: list,
            n_env_conditions: int, plastic_body: int, plastic_brain: int, loop: str, body_phenotype: str, bisymmetry: int) -> ModularRobot:
    
    # TODO: ugly rules to overcome revolves low modularity
    #  additionally, im treating cppnwin as a source of representations.

    # TODO: closed loop with evolvable body does not work yet (ann expects spider inputs and outputs)

    if body_phenotype == 'evolvable':
        # body, queried_substrate = cppn_body_develop(max_modules, substrate_radius, genotype.body, querying_seed,
        #                     env_condition, n_env_conditions, plastic_body, bisymmetry).develop()
    
        body, queried_substrate = grn_develop(max_modules, genotype.body, querying_seed,
                                                   env_condition, n_env_conditions, plastic_body, bisymmetry).develop()

    if body_phenotype == 'spider':
        body = make_body_spider()
        queried_substrate = []

    ###

    # cpg
    if loop == 'open':
        brain = brain_cpg_develop(genotype.brain, body, env_condition, n_env_conditions, plastic_brain)

    # proprioceptive ann
    if loop == 'closed':
        brain = brain_ann_develop(genotype.brain, body, env_condition, n_env_conditions, plastic_brain)

    return ModularRobot(body, brain), queried_substrate


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