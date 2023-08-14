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

    genome_size = 100
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
        self.expected_genes = 10
        self.concentration_decay = 0.005
        # tfs = (B, A1, A2)
        self.structural_trs = 3
        self.regulatory_tfs = math.ceil(self.structural_trs * 2.5)
        self.increase_scaling = 100
        self.intra_diffusion_prop = 1/2
        self.inter_diffusion_prop = self.intra_diffusion_prop/8
        self.dev_steps = 1#500

    def develop(self):

        self.random = random.Random(self.querying_seed)
        self.quantity_nodes = 0

        self.develop_body()
        self.phenotype_body.finalize()

        return self.phenotype_body, self.queried_substrate

    def develop_body(self):
        print('\n\n')
        self.gene_parser()
        self.regulate()

        return self.phenotype_body

    # parses genotype to discover promotor sites and mother tf injection
    def gene_parser(self):

        promoter_threshold = self.expected_genes/self.pop_size
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
                    #print(regulatory_transcription_factor, regulatory_min, regulatory_max,
                     #        transcription_factor, transcription_factor_amount, diffusion_site)
                    # begin: converts tfs #
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

                    # ends: converts tfs #

                    # begin: converts diffusion sites #
                    range_size = 1 / 4
                    limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]

                    for idx in range(0, len(limits) - 1):
                        if limits[idx+1] > diffusion_site >= limits[idx]:
                            diffusion_site_label = idx
                        elif diffusion_site >= limits[idx+1]:
                            diffusion_site_label = len(limits)-1

                    # ends: converts diffusion sites #

                    genes = [regulatory_transcription_factor_label, regulatory_min, regulatory_max,
                             transcription_factor_label, transcription_factor_amount, diffusion_site_label]

                    if len(self.promotors) == 0:
                        self.promotors = np.hstack((self.promotors, np.array(genes)))
                    else:
                        self.promotors = np.vstack((self.promotors, np.array(genes)))
                    nucleotide_idx += len(genes)

            nucleotide_idx += 1
        pprint.pprint(self.promotors)

    def regulate(self):
        first_cell = self.maternal_injection()
        self.growth(first_cell)

    def growth(self, cell):
        print('\ngrowth')
        for t in range(0, self.dev_steps):
            for tf in cell.transcription_factors:
                print(t, tf)

                print('inc')
                # increase
                tf_promotors = np.where(self.promotors[:, 3] == tf)[0]
                for tf_promotor_idx in tf_promotors:
                    cell.transcription_factors[tf][int(self.promotors[tf_promotor_idx][5])] += \
                    float(self.promotors[tf_promotor_idx][4]) / float(self.increase_scaling)
                    print(cell.transcription_factors[tf])

                # intra diffusion
                print('intra')
                intra_diffusion_rate = self.concentration_decay/self.intra_diffusion_prop

                # inter diffusion
                print('inter')
                inter_diffusion_rate = self.concentration_decay/self.inter_diffusion_prop

                # decay
                print('dec')
                cell.transcription_factors[tf] = list(np.array(cell.transcription_factors[tf])-self.concentration_decay)
                print(cell.transcription_factors[tf])

    def maternal_injection(self):

        # injects maternal tf into single cell embryo and starts development of the first cell
        mother_tf_label = self.promotors[0][0]
        mother_tf_injection = float(self.promotors[0][1])

        first_cell = Cell()
        self.embryo = first_cell
        first_cell.developed_cell = self.place_head(first_cell)
        first_cell.transcription_factors[mother_tf_label] = [mother_tf_injection/4]*4
        print('\n head', mother_tf_label, mother_tf_injection)
        self.express_promoters(first_cell)
        return first_cell

    def express_promoters(self, new_cell):

        regulatory_transcription_factor = 0
        regulatory_min = 1
        regulatory_max = 2
        transcription_factor = 3
        transcription_factor_amount = 4
        diffusion_site = 5

        for promotor in self.promotors:

            regulatory_min_val = min(float(promotor[regulatory_min]), float(promotor[regulatory_max]))
            regulatory_max_val = max(float(promotor[regulatory_min]), float(promotor[regulatory_max]))
            print(promotor[regulatory_transcription_factor])
            # expresses if regulatory tf is present and within range
            if new_cell.transcription_factors.get(promotor[regulatory_transcription_factor]) \
                 and sum(new_cell.transcription_factors[promotor[regulatory_transcription_factor]]) >= regulatory_min_val \
                 and sum(new_cell.transcription_factors[promotor[regulatory_transcription_factor]]) <= regulatory_max_val:

                if new_cell.transcription_factors.get(promotor[transcription_factor]):
                    new_cell.transcription_factors[promotor[transcription_factor]][int(promotor[diffusion_site])] += \
                        float(promotor[transcription_factor_amount])
                else:
                    new_cell.transcription_factors[promotor[transcription_factor]] = [0]*4
                    new_cell.transcription_factors[promotor[transcription_factor]][int(promotor[diffusion_site])] =\
                        float(promotor[transcription_factor_amount])
            print('--')
            for t in new_cell.transcription_factors:
                print(t, new_cell.transcription_factors[t])

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