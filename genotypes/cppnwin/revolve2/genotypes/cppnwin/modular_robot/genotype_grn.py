import math
import multineat
import random
import numpy as np
import operator
import sys
import pprint

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, Core, Module
from .._genotype import Genotype
from .._random_v1 import random_v1 as base_random_v1


def random_v1(
    rng
) -> Genotype:

    genome_size = 20+2 #100+2

    genotype = [round(rng.uniform(0, 1), 2) for _ in range(genome_size)]
    return Genotype(genotype)


class Develop:

    def __init__(self, max_modules, genotype, querying_seed, env_condition, n_env_conditions, plastic_body, bisymmetry):

        self.max_modules = max_modules
        self.genotype = genotype.genotype
        self.querying_seed = querying_seed
        self.env_condition = env_condition
        self.n_env_conditions = n_env_conditions
        self.plastic_body = plastic_body
        self.bisymmetry = bisymmetry
        self.random = None
        self.queried_substrate = {}
        self.embryo = []
        self.phenotype_body = None
        self.promotors = np.array([])
        self.quantity_modules = 0

        self.pop_size = 100 # remember to match with popsize in config
        self.expected_genomes_n = 70 # 10
        self.concentration_decay = 0.005
        # tfs = (B, A1, A2)
        self.structural_trs = 3
        self.regulatory_tfs = math.ceil(self.structural_trs * 2.5)
        self.increase_scaling = 100
        self.inter_unit_diffusion = 1/2
        self.intra_unit_diffusion = self.inter_unit_diffusion/8
        self.dev_steps = 500

    def develop(self):

        self.random = random.Random(self.querying_seed)
        self.quantity_nodes = 0

        self.develop_body()
        self.phenotype_body.finalize()

        return self.phenotype_body, self.queried_substrate

    def develop_body(self):
        print('\n\n')
        mother_tf_label, mother_tf_injection = self.gene_parser()
        self.regulate(mother_tf_label, mother_tf_injection)

        return self.phenotype_body

    # parses genotype to discover promotor sites and mother tf injection
    def gene_parser(self):

        promoter_threshold = self.expected_genomes_n/self.pop_size

        mother_tf = self.genotype[0]
        mother_tf_injection = self.genotype[1]
        self.genotype = self.genotype[2:]

        nucleotide_idx = 0
        while nucleotide_idx < len(self.genotype):

            if self.genotype[nucleotide_idx] < promoter_threshold:
                if (len(self.genotype)-1-nucleotide_idx) >= 6:

                    regulatory_transcription_factor = self.genotype[nucleotide_idx+1]  # gene product
                    regulatory_min = self.genotype[nucleotide_idx+2]
                    regulatory_max = self.genotype[nucleotide_idx+3]
                    transcription_factor = self.genotype[nucleotide_idx+4]
                    transcription_factor_amount = self.genotype[nucleotide_idx+5]
                    diffusion_site = self.genotype[nucleotide_idx+6]

                 #   print('\n--- promotor gene', regulatory_transcription_factor,regulatory_min,
                   #       regulatory_max,transcription_factor,transcription_factor_amount,diffusion_site,'\n')

                    range_size = 1 / (self.structural_trs + self.regulatory_tfs)
                    limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]

                    for idx in range(0, len(limits)-1):

                        if regulatory_transcription_factor >= limits[idx] and regulatory_transcription_factor < limits[idx+1]:
                            regulatory_transcription_factor_label = 'TF'+str(idx+1)
                        elif regulatory_transcription_factor >= limits[idx + 1]:
                                regulatory_transcription_factor_label = 'TF' + str(len(limits))

                        if transcription_factor >= limits[idx] and transcription_factor < limits[idx+1]:
                            transcription_factor_label = 'TF'+str(idx+1)
                        elif transcription_factor >= limits[idx+1]:
                            transcription_factor_label = 'TF'+str(len(limits)-1)

                        if mother_tf >= limits[idx] and mother_tf < limits[idx + 1]:
                            mother_tf_label = 'TF' + str(idx + 1)
                        elif mother_tf >= limits[idx+1]:
                             mother_tf_label = 'TF'+str(len(limits))

                    genes = [regulatory_transcription_factor_label, regulatory_min, regulatory_max,
                                     transcription_factor_label, transcription_factor_amount, diffusion_site]

                    if len(self.promotors) == 0:
                        self.promotors = np.hstack((self.promotors, np.array(genes)))
                    else:
                        self.promotors = np.vstack((self.promotors, np.array(genes)))
                    nucleotide_idx += len(genes)

            nucleotide_idx += 1
        pprint.pprint(self.promotors)
        return mother_tf_label, mother_tf_injection

    def regulate(self, other_tf_label, mother_tf_injection):
        self.maternal_injection(other_tf_label, mother_tf_injection)

    def attach_body(self):
        pass

    def maternal_injection(self, mother_tf_label, mother_tf_injection):

        # injects maternal tf into single cell embryo and starts development of the first cell
        new_cell = Cell()
        self.embryo = new_cell
        new_cell.developed_cell = self.place_head(new_cell)
        new_cell.transcription_factors[mother_tf_label] = mother_tf_injection
        print('\n head')
        pprint.pprint(new_cell.transcription_factors)
        self.express_promoters(new_cell)
        pprint.pprint(new_cell.transcription_factors)

    def express_promoters(self, new_cell):
        pass

    def place_head(self, new_cell):

        module_type = Core
        self.phenotype_body = Body()
        self.phenotype_body.core._id = self.quantity_modules
        orientation = 0
        self.phenotype_body.core._rotation = orientation * (math.pi / 2.0)
        self.phenotype_body.core._orientation = 0
        self.phenotype_body.core.rgb = self.get_color(module_type, orientation)
        self.phenotype_body.core.substrate_coordinates = (0, 0)
        self.queried_substrate[(0, 0)] = self.phenotype_body.core
        self.phenotype_body.core.developed_cell = new_cell

        return self.phenotype_body.core

    def get_color(self, module_type, rotation):
        rgb = []
        if module_type == Brick:
            rgb = [0, 0, 1]
        if module_type == ActiveHinge:
            if rotation == 0:
                rgb = [1, 0.08, 0.58]
            else:
                rgb = [0.7, 0, 0]
        if module_type == Core:
            rgb = [1, 1, 0]
        return rgb


class Cell:

    def __init__(self) -> None:
        self.developed_cell = None
        self.transcription_factors = {}