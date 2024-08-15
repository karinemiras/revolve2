import pandas as pd
import argparse
import warnings
import numpy as np

import plotly.graph_objs as go
import plotly.offline as offline
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("study")
parser.add_argument("experiments")
parser.add_argument("tfs")
parser.add_argument("watchruns")
parser.add_argument("generations")
parser.add_argument("mainpath")
args = parser.parse_args()

study = args.study
experiments_name = args.experiments.split(',')
tfs = list(args.tfs.split(','))
generations = [100]
mainpath = args.mainpath

path = f'{mainpath}/{study}/analysis/knockouts/data'


def positive_and_zero(a, b):
    return (a > 0 and b == 0) or (a == 0 and b > 0)


def negative_and_zero(a, b):
    return (a < 0 and b == 0) or (a == 0 and b < 0)


def positive_positive(a, b):
    return a > 0 and b > 0


def negative_negative(a, b):
    return a < 0 and b < 0


def calculate_general():
    origin_file = f'{path}/knockouts_measures.csv'
    df_ori = pd.read_csv(origin_file)
    original = 'o'  # original phenotype, without knockout

    keys = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id']
    traits = ['disp_y', 'distance', 'symmetry', 'extremities_prop']
    others = ['knockout']
    df = df_ori.filter(items=keys + others + traits)

    # df = df[   ( ( df['experiment_name'] == 'reg2m2') & (df['run'] == 1) &  (df['gen'] == 0 )  ) ] # quick test

    df = df[((df['gen'] == generations[0]) | (df['gen'] == generations[-1]))]

    for trait in traits:
        # sends trait values of each knockout to columns
        pivot_df = df.pivot_table(index=keys,
                                  columns='knockout', values=trait,
                                  # for distance variable, which is not a trait,
                                  # the calculation is idle (compared to 0)
                                  aggfunc='first')

        all_columns = pivot_df.columns
        knock_columns = [col for col in all_columns if col not in keys and col != original]

        # Subtract each knock_columns by the original
        # (positive values mean the mutant had an increase in the trait or growth)
        df_delta = pivot_df.drop(columns=original).sub(pivot_df[original], axis=0)

        double_knocks = [col for col in knock_columns if '.' in col]
        for mxy in double_knocks:
            genes = mxy.split('.')

            mx = genes[0]
            my = genes[1]

            df_delta[f'{mx}categ{my}'] = ''

            df_delta[f'{mx}categ{my}'] = df_delta.apply(
                lambda row: 'buffering'
                if (row[mx] == 0 and row[my] == 0 and row[mxy] > 0) or \
                   (row[mx] == 0 and row[my] == 0 and row[mxy] < 0)
                else row[f'{mx}categ{my}'], axis=1
            )

            df_delta[f'{mx}categ{my}'] = df_delta.apply(
                lambda row: 'supression'
                if (positive_positive(row[mx], row[my]) and row[mxy] == 0) or \
                   (positive_and_zero(row[mx], row[my]) and row[mxy] == 0) or \
                   (negative_negative(row[mx], row[my]) and row[mxy] == 0) or \
                   (negative_and_zero(row[mx], row[my]) and row[mxy] == 0)
                else row[f'{mx}categ{my}'], axis=1
            )

            df_delta[f'{mx}categ{my}'] = df_delta.apply(
                lambda row: 'quantitative_buffering'
                if (positive_positive(row[mx], row[my]) and row[mxy] > row[mx] and row[mxy] > row[my]) or \
                   (positive_and_zero(row[mx], row[my]) and row[mxy] > row[mx] and row[mxy] > row[my]) or \
                   (negative_negative(row[mx], row[my]) and row[mxy] < row[mx] and row[mxy] < row[my]) or \
                   (negative_and_zero(row[mx], row[my]) and row[mxy] < row[mx] and row[mxy] < row[my])
                else row[f'{mx}categ{my}'], axis=1
            )

            df_delta[f'{mx}categ{my}'] = df_delta.apply(
                lambda row: 'quantitative_supression'
                if (positive_positive(row[mx], row[my]) and row[mxy] < row[mx] and row[mxy] < row[my] and row[
                    mxy] > 0) or \
                   (positive_and_zero(row[mx], row[my]) and row[mxy] < row[mx] and row[mxy] < row[my] and row[
                       mxy] > 0) or \
                   (negative_negative(row[mx], row[my]) and row[mxy] > row[mx] and row[mxy] > row[my] and row[
                       mxy] < 0) or \
                   (negative_and_zero(row[mx], row[my]) and row[mxy] > row[mx] and row[mxy] > row[my] and row[mxy] < 0)
                else row[f'{mx}categ{my}'], axis=1
            )

            df_delta[f'{mx}categ{my}'] = df_delta.apply(
                lambda row: 'masking'
                if (((row[mx] > 0 and row[my] < 0) or (row[mx] < 0 and row[my] > 0)) \
                    and row[mxy] > 0) or \
                   (((row[mx] > 0 and row[my] < 0) or (row[mx] < 0 and row[my] > 0)) \
                    and row[mxy] < 0)
                else row[f'{mx}categ{my}'], axis=1
            )

            df_delta[f'{mx}categ{my}'] = df_delta.apply(
                lambda row: 'inversion'
                if (positive_positive(row[mx], row[my]) and row[mxy] < 0) or \
                   (positive_and_zero(row[mx], row[my]) and row[mxy] < 0) or \
                   (negative_negative(row[mx], row[my]) and row[mxy] > 0) or \
                   (negative_and_zero(row[mx], row[my]) and row[mxy] > 0)
                else row[f'{mx}categ{my}'], axis=1
            )

        categ_columns = [col for col in df_delta.columns if 'categ' in col]

        buffering = df_delta[categ_columns] == 'buffering'
        supression = df_delta[categ_columns] == 'supression'
        quantitative_buffering = df_delta[categ_columns] == 'quantitative_buffering'
        quantitative_supression = df_delta[categ_columns] == 'quantitative_supression'
        masking = df_delta[categ_columns] == 'masking'
        inversion = df_delta[categ_columns] == 'inversion'

        count_buffering = buffering.sum(axis=1)
        count_supression = supression.sum(axis=1)
        count_quantitative_buffering = quantitative_buffering.sum(axis=1)
        count_quantitative_supression = quantitative_supression.sum(axis=1)
        count_masking = masking.sum(axis=1)
        count_inversion = inversion.sum(axis=1)

        df_delta['buffering'] = count_buffering
        df_delta['supression'] = count_supression
        df_delta['quantitative_buffering'] = count_quantitative_buffering
        df_delta['quantitative_supression'] = count_quantitative_supression
        df_delta['masking'] = count_masking
        df_delta['inversion'] = count_inversion

        df_exp = df_delta.reset_index()[
            keys + ['buffering', 'supression', 'quantitative_buffering', 'quantitative_supression', 'masking',
                    'inversion']]
        df_exp.to_csv(f'{path}/knockouts/data/effectscateg_{trait}.csv')

        print(trait)


calculate_general()


