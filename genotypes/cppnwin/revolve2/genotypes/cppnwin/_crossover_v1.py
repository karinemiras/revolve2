import sys

from ._genotype import Genotype


def crossover_v1(
    parent1,
    parent2,
    rng,
) -> Genotype:

    # TODO: this threshold and types  should match with the develop method automatically
    promoter_threshold = 0.8
    types_nucleotypes = 6

    # the first nucleotide is the concentation
    new_genotype = [(parent1.genotype[0]+parent2.genotype[0])/2]
    p1 = parent1.genotype[1:]
    p2 = parent2.genotype[1:]
   # print('\nbefore')
   # print('parent1 ',parent1.genotype)
   # print('parent2 ', parent2.genotype)

    for parent in [p1, p2]:
      #  print('\nparent', parent)
        nucleotide_idx = 0
        promotor_sites = []
        while nucleotide_idx < len(parent):
            if parent[nucleotide_idx] < promoter_threshold:
                # if there are nucleotypes enough to compose a gene
                if (len(parent)-1-nucleotide_idx) >= types_nucleotypes:
                   # print('gene',nucleotide_idx, parent[nucleotide_idx], ' - ', parent[nucleotide_idx+1], parent[nucleotide_idx+2], parent[nucleotide_idx+3], parent[nucleotide_idx+4], parent[nucleotide_idx+5], parent[nucleotide_idx+6])
                    promotor_sites.append(nucleotide_idx)
                    nucleotide_idx += types_nucleotypes
            nucleotide_idx += 1
       # print(promotor_sites)
        cutpoint = rng.sample(promotor_sites, 1)[0]
       # print('cutpoint',cutpoint)
        subset = parent[0:cutpoint+types_nucleotypes+1]
        new_genotype += subset
       # print('subset', subset)
    #print('offspring',new_genotype)

    # print('parent after')
    # print('parent1', parent1.genotype)
    # print('parent2', parent2.genotype)
    # new_genotype = new_genotype[1:]
    # nucleotide_idx = 0
    # while nucleotide_idx < len(new_genotype):
    #     if new_genotype[nucleotide_idx] < promoter_threshold:
    #         if (len(new_genotype) - 1 - nucleotide_idx) >= types_nucleotypes:
    #             print('gene', nucleotide_idx, new_genotype[nucleotide_idx], ' - ', new_genotype[nucleotide_idx + 1],
    #                   new_genotype[nucleotide_idx + 2], new_genotype[nucleotide_idx + 3], new_genotype[nucleotide_idx + 4],
    #                   new_genotype[nucleotide_idx + 5], new_genotype[nucleotide_idx + 6])
    #             nucleotide_idx += types_nucleotypes
    #     nucleotide_idx += 1
    # sys.exit()
    return Genotype(new_genotype)
