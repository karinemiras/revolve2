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
        self.cells = []
        self.phenotype_body = None
        self.promotors = np.array([])
        self.quantity_modules = 0

        self.regulatory_transcription_factor_idx = 0
        self.regulatory_min_idx = 1
        self.regulatory_max_idx = 2
        self.transcription_factor_idx = 3
        self.transcription_factor_amount_idx = 4
        self.diffusion_site_idx = 5
        self.types_nucleotypes = 6
        self.diffusion_sites_qt = 4

        self.pop_size = 100 # remember to match with popsize in config
        self.expected_genes = 10
        self.concentration_decay = 0.005
        self.structural_trs = 3 # tfs = (B, A1, A2) # the last three
        self.regulatory_tfs = math.ceil(self.structural_trs * 2.5)
        self.increase_scaling = 100
        self.intra_diffusion_rate = self.concentration_decay/2
        self.inter_diffusion_rate  = self.intra_diffusion_rate/8
        self.dev_steps = 20#500
        self.concentration_threshold = 0.1

    def develop(self):

        self.random = random.Random(self.querying_seed)
        self.quantity_nodes = 0
        self.develop_body()
        self.phenotype_body.finalize()

        return self.phenotype_body, {}

    def develop_body(self):
        print('\n\n')
        self.gene_parser()
        self.regulate()

        return self.phenotype_body

    # parses genotype to discover promotor sites and compose genes
    def gene_parser(self):

        promoter_threshold = self.expected_genes/self.pop_size
        nucleotide_idx = 0
        while nucleotide_idx < len(self.genotype):

            if self.genotype[nucleotide_idx] < promoter_threshold:
                # if there are nucleotypes enough to compose a gene
                if (len(self.genotype)-1-nucleotide_idx) >= self.types_nucleotypes:
                    regulatory_transcription_factor = self.genotype[nucleotide_idx+self.regulatory_transcription_factor_idx+1]  # gene product
                    regulatory_min = self.genotype[nucleotide_idx+self.regulatory_min_idx+1]
                    regulatory_max = self.genotype[nucleotide_idx+self.regulatory_max_idx+1]
                    transcription_factor = self.genotype[nucleotide_idx+self.transcription_factor_idx+1]
                    transcription_factor_amount = self.genotype[nucleotide_idx+self.transcription_factor_amount_idx+1]
                    diffusion_site = self.genotype[nucleotide_idx+self.diffusion_site_idx+1]

                    # begin: converts tfs values into labels #
                    range_size = 1 / (self.structural_trs + self.regulatory_tfs)
                    limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]
                    for idx in range(0, len(limits)-1):

                        if regulatory_transcription_factor >= limits[idx] and regulatory_transcription_factor < limits[idx+1]:
                            regulatory_transcription_factor_label = 'TF'+str(idx+1)
                        elif regulatory_transcription_factor >= limits[idx+1]:
                            regulatory_transcription_factor_label = 'TF' + str(len(limits))

                        if transcription_factor >= limits[idx] and transcription_factor < limits[idx+1]:
                            transcription_factor_label = 'TF'+str(idx+1)
                        elif transcription_factor >= limits[idx+1]:
                            transcription_factor_label = 'TF'+str(len(limits)-1)
                    # ends: converts tfs values into labels #

                    # begin: converts diffusion sites values into labels #
                    range_size = 1 / self.diffusion_sites_qt
                    limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]
                    for idx in range(0, len(limits) - 1):
                        if limits[idx+1] > diffusion_site >= limits[idx]:
                            diffusion_site_label = idx
                        elif diffusion_site >= limits[idx+1]:
                            diffusion_site_label = len(limits)-1
                    # ends: converts diffusion sites values into labels #

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
        self.maternal_injection()
        self.growth()

    def growth(self):
        print('\ngrowth')
        for t in range(0, self.dev_steps):
            print('----------------t',t)
            for idxc in range(0, len(self.cells)):
                print('---cell', idxc)
                cell = self.cells[idxc]
                for tf in cell.transcription_factors:
                   # print(' ', tf)
                    self.increase(tf, cell)
                    self.intra_diffusion(tf, cell)
                    self.inter_diffusion(tf, cell)
                    self.decay(tf, cell)

                print('\n place module')
                self.place_module(cell)

            #
            # for tf in cell.transcription_factors:
            #     print(cell.transcription_factors[tf])
            #
    def increase(self, tf, cell):

        # increase concentration in due diffusion sites
        tf_promotors = np.where(self.promotors[:, self.transcription_factor_idx] == tf)[0]
        for tf_promotor_idx in tf_promotors:
            cell.transcription_factors[tf][int(self.promotors[tf_promotor_idx][self.diffusion_site_idx])] += \
                float(self.promotors[tf_promotor_idx][self.transcription_factor_amount_idx]) \
                / float(self.increase_scaling)

    def inter_diffusion(self, tf, cell):
        # inter diffusion
        pass
    def intra_diffusion(self, tf, cell):

        # intra diffusion in all sites: clockwise and right-left
        for ds in range(0, self.diffusion_sites_qt):
            # print(' ds', ds)
            ds_left = ds - 1 if ds - 1 >= 0 else self.diffusion_sites_qt - 1
            ds_right = ds + 1 if ds + 1 <= self.diffusion_sites_qt - 1 else 0

            if cell.transcription_factors[tf][ds] >= self.intra_diffusion_rate:
                cell.transcription_factors[tf][ds] -= self.intra_diffusion_rate
                cell.transcription_factors[tf][ds_right] += self.intra_diffusion_rate

            if cell.transcription_factors[tf][ds] >= self.intra_diffusion_rate:
                cell.transcription_factors[tf][ds] -= self.intra_diffusion_rate
                cell.transcription_factors[tf][ds_left] += self.intra_diffusion_rate

    def decay(self, tf, cell):
        # decay in all sites
        for ds in range(0, self.diffusion_sites_qt):
            cell.transcription_factors[tf][ds] = \
                max(0, cell.transcription_factors[tf][ds] - self.concentration_decay)
        print(tf, cell.transcription_factors[tf], sum(cell.transcription_factors[tf]))

    def place_module(self, cell):

        product_tfs = ['TF9', 'TF10', 'TF11']
        modules_types = [Brick, ActiveHinge, ActiveHinge]  # fix A rotation later

        concentration1 = sum(cell.transcription_factors[product_tfs[0]]) \
            if cell.transcription_factors.get(product_tfs[0]) else 0  # B

        concentration2 = sum(cell.transcription_factors[product_tfs[1]]) \
            if cell.transcription_factors.get(product_tfs[1]) else 0  # A1

        concentration3 = sum(cell.transcription_factors[product_tfs[2]]) \
            if cell.transcription_factors.get(product_tfs[2]) else 0  # A2

        product_concentrations = [concentration1, concentration2, concentration3]
        idx_max = product_concentrations.index(max(product_concentrations))

        print(product_concentrations, idx_max)
        # tries fo grow the product with the highest concentration if above a threshold
        # grows in the free diffusion site with the highest concentration
        if product_concentrations[idx_max] > self.concentration_threshold:

            # TODO: and slot free
            slot = cell.transcription_factors[product_tfs[idx_max]]. \
                index(max(cell.transcription_factors[product_tfs[idx_max]]))
            new_module = None
            print('type',module_type, slot)

            module_type = modules_types[idx_max]
            self.new_cell(product_tfs[idx_max], cell, new_module, slot)

    def new_cell(self, tf, cell, new_module, slot):
        print('new')
        cell.transcription_factors[tf][slot] = cell.transcription_factors[tf][slot]/2

        print(tf, cell.transcription_factors[tf], sum(cell.transcription_factors[tf]))

    def maternal_injection(self):

        # injects maternal tf into single cell embryo and starts development of the first cell
        # the tf injected is regulatory tf of the first gene in the genetic string
        # the amount inject is the minimum for the regulatory tf to regulate its regulated product
        first_gene_idx = 0
        tf_label_idx = 0
        min_value_idx = 1
        mother_tf_label = self.promotors[first_gene_idx][tf_label_idx]
        mother_tf_injection = float(self.promotors[first_gene_idx][min_value_idx])

        first_cell = Cell()
        # distributes injection among diffusion sites
        first_cell.transcription_factors[mother_tf_label] = \
            [mother_tf_injection/self.diffusion_sites_qt] * self.diffusion_sites_qt
        print('\n head', mother_tf_label, mother_tf_injection)
        self.express_promoters(first_cell)
        self.cells.append(first_cell)
        first_cell.developed_cell = self.place_head(first_cell)


    def express_promoters(self, new_cell):

        for promotor in self.promotors:

            regulatory_min_val = min(float(promotor[self.regulatory_min_idx]),
                                     float(promotor[self.regulatory_max_idx]))
            regulatory_max_val = max(float(promotor[self.regulatory_min_idx]),
                                     float(promotor[self.regulatory_max_idx]))
            print(promotor[self.regulatory_transcription_factor_idx])
            # expresses a tf if its regulatory tf is present and within a range
            if new_cell.transcription_factors.get(promotor[self.regulatory_transcription_factor_idx]) \
                 and sum(new_cell.transcription_factors[promotor[self.regulatory_transcription_factor_idx]]) \
                    >= regulatory_min_val \
                 and sum(new_cell.transcription_factors[promotor[self.regulatory_transcription_factor_idx]]) \
                    <= regulatory_max_val:

                if new_cell.transcription_factors.get(promotor[self.transcription_factor_idx]):
                    new_cell.transcription_factors[promotor[self.transcription_factor_idx]] \
                        [int(promotor[self.diffusion_site_idx])] +=  float(promotor[self.transcription_factor_amount_idx])
                else:
                    new_cell.transcription_factors[promotor[self.transcription_factor_idx]] = [0] * self.diffusion_sites_qt
                    new_cell.transcription_factors[promotor[self.transcription_factor_idx]] \
                    [int(promotor[self.diffusion_site_idx])] = float(promotor[self.transcription_factor_amount_idx])
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