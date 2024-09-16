import polars as pl
from useful_functions import (read_columns, \
                              remove_missing_variant_ids_from_indices,
                              compute_ldm_chunks,
                              compute_prs_statistics, join_ldm_chunks)
from constants import *
from utils import run, load_dbSNP, get_rs_numbers_bim_or_pvar, \
    make_bim_or_pvar_IDs_unique, reverse_make_bim_or_pvar_IDs_unique
from multiprocessing import Process, Pool
import logging
import sys, os

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# TODO: When re-running, compute 'BETA' via log(OR) instead of joining on
#  location and ref + alt alleles, where OR is provided in munged file.
def transfer_rsids(summary_stats_file_munged: str,
                   summary_stats_file_raw: str,
                   save_to: str) -> None:
    """
    Transfers RS IDs from a munged summary stats file to the raw file. Used
    to obtain betas.
    """
    summary_stats_raw = read_columns(summary_stats_file_raw, delimiter='\t',
                                     data_types={'CHROM': pl.String})
    summary_stats_munged = read_columns(summary_stats_file_munged,
                                        delimiter='\t',
                                        data_types={'CHROM': pl.String})
    summary_stats_munged = summary_stats_munged.with_columns(
        CHROM=pl.col('CHROM').str.slice(offset=3))
    summary_stats_munged = summary_stats_munged.join(
        summary_stats_raw.select('CHROM', 'POS', 'A1', 'A2', 'BETA'),
        how='left',
        left_on=['CHROM', 'BP',
                 'REF', 'ALT'],
        right_on=['CHROM',
                  'POS', 'A1',
                  'A2'],
        coalesce=True)
    summary_stats_munged.collect().write_csv(save_to, separator='\t')


def convert_summary_stats_to_megaprs(summary_stats_file: str,
                                     save_to: str) -> None:
    """
    Converts summary stats file to the following format - Predictor | A1 | A2
    | n | Z (beta scaled by standard deviation).
    :param summary_stats_file: path to the summary statistics file
    :param save_to: path to save the converted summary statistics file to
    """
    summary_stats = read_columns(summary_stats_file, None, delimiter='\t',
                                 data_types={'CHROM': pl.String,
                                             'BETA': pl.Float64,
                                             'SE': pl.Float64})
    summary_stats = summary_stats.with_columns(
        Z=pl.col('BETA') / pl.col('SE'))
    summary_stats = make_summary_statistics_unique(summary_stats, 'SNP',
                                                   'CHROM',
                                                   'ALT', 'REF')
    summary_stats = summary_stats.select(Predictor='SNP', A1='REF', A2='ALT',
                                         n='N', Z='Z').filter(
        pl.col('Predictor').is_not_null())
    summary_stats.collect().write_csv(save_to, separator='\t')


def convert_summary_stats_to_gcta(summary_stats_file: str,
                                  save_to: str) -> None:
    """
    Converts summary stats file to the following format - SNP | A1 | A2 | freq(
    uency of effect
    allele) | b(eta) | se (standard error) | p(-value) | N (sample size).
    :param summary_stats_file: path to the summary statistics file
    :param save_to: path to save the converted summary statistics file to
    """
    summary_stats = read_columns(summary_stats_file, None, delimiter='\t')
    summary_stats = summary_stats.with_columns(freq=
                                               (pl.col('FCAS') * pl.col(
                                                   'NCAS') + pl.col(
                                                   'FCON') * pl.col('NCON')) / (
                                                       pl.col('NCAS') + pl.col(
                                                   'NCON')),
                                               N=pl.col('NEFFDIV2') * 2)
    summary_stats = summary_stats.select(SNP='ID', A1='A1', A2='A2',
                                         freq='freq', b='BETA', se='SE',
                                         p='PVAL', N='N')
    summary_stats.collect().write_csv(save_to, separator='\t')


def compute_high_ld_region(genome_file: str, high_ld_regions: str,
                           save_to: str) -> None:
    """
    Computes high LD regions in megaprs required format from appropriate file.
    """
    run(f'{LDAK} '
        f'--cut-genes {save_to} '
        f'--bfile {genome_file} '
        f'--genefile {high_ld_regions}')


def assign_scores(scores_file: str, genome_file: str, save_to: str) -> None:
    """
    Assigns PRS scores to individuals.

    Pre-conditions:
        - Run compute_prs_statistics for scores.
    """
    run(f'{LDAK} '
        f'--calc-scores {save_to} '
        f'--bfile {genome_file} '
        f'--scorefile {scores_file} '
        f'--power 0 ')


# TODO: there's probably an issue with reformatting.
def reformat_high_ld_regions(high_ld_regions_file: str, save_to: str) -> None:
    """
    Reformats high LD regions file to megaprs accepted format.
    """
    # Read file and split up range into struct containing start and end SNP.
    high_ld_regions = (read_columns(high_ld_regions_file, delimiter='\t',
                                    column_names=['chr', 'start', 'end',
                                                  'range', 'misc.'])
    .with_columns(
        range=pl.col('range')
        .str
        .split_exact(pl.lit(':'), 1)
        .struct.field('field_1')
        .str
        .split_exact(pl.lit('-'), 1),
        chr=pl.col('chr')
        .str
        .slice(3)
        .str
        .to_integer()
    ))

    # Create start and end columns.
    high_ld_regions = (high_ld_regions
    .with_columns(
        start=pl.col('range').struct.field('field_0'),
        end=pl.col('range').struct.field('field_1')))

    # Sort within chromosome by start position.
    high_ld_regions = high_ld_regions.sort(pl.col('chr'), pl.col('start'),
                                           descending=[False, False])
    high_ld_regions.collect().write_csv(save_to, separator='\t')


def make_summary_statistics_unique(summary_statistics: pl.LazyFrame,
                                   id_col: str,
                                   chrom_col: str,
                                   effect_allele_col: str,
                                   reference_allele_col: str) -> pl.LazyFrame:
    """
    Uniquifies summary statistics' RS IDs. Renaming strategy is the following -
    concat with chrom, ref, alt if an RS ID, else add chrom between ID and ref
    (chrom:ID_ref_alt format previously). Probably redundant now, but was
    used previously.
    """
    # Join with ~.
    args = {id_col: pl.concat_str(pl.col(id_col), pl.col(chrom_col),
                                  pl.col(reference_allele_col),
                                  pl.col(effect_allele_col),
                                  separator='_')}

    summary_statistics = summary_statistics.with_columns(**args)

    return summary_statistics


def extract_phenotype_data(metadata_file: str, id_col: str, phenotype_col: str,
                           save_to: str) -> None:
    """
    Writes case-control status and IDs to another file in the FID | IID |
    schizophrenia_status format.
    """
    metadata = read_columns(metadata_file, delimiter='\t')
    metadata = metadata.select(FID=id_col, IID=id_col,
                               schizophrenia_status=phenotype_col)
    metadata.collect().write_csv(save_to, separator='\t')


if __name__ == '__main__':
    ### FORMATTING FIXES ###
    if not os.path.exists(
            f'{SCHIZO_DIR_GWAS}/schizo_gwas_summary_stats.megaprs'):
        transfer_rsids(f'{SCHIZO_DIR_GWAS}/schizo_gwas_summary_stats_munged',
                       f'{SCHIZO_DIR_GWAS}/schizo_gwas_summary_stats_cleaned',
                       f'{SCHIZO_DIR_GWAS}/schizo_gwas_summary_stats')
        convert_summary_stats_to_megaprs(
            f'{SCHIZO_DIR_GWAS}/schizo_gwas_summary_stats',
            f'{SCHIZO_DIR_GWAS}/schizo_gwas_summary_stats.megaprs')

    if not os.path.exists(
            f'{PROJECT_DIR}/jeesonja/high_ld_regions_reformatted'):
        reformat_high_ld_regions(f'{PROJECT_DIR}/jeesonja/high_ld_regions.txt',
                                 f'{PROJECT_DIR}/jeesonja/high_ld_regions_reformatted')

    ### IMPORTANT COVARIATES ###
    # TODO: Try looking into this - exclude high LD region.
    if not os.path.exists(F'{SCHIZO_DIR_PRS}/mclean_high_ld_regions'):
        compute_high_ld_region(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc',
            f'{PROJECT_DIR}/jeesonja/high_ld_regions_reformatted.txt',
            F'{SCHIZO_DIR_PRS}/mclean_high_ld_regions')

    if not os.path.exists(f'{SCHIZO_DIR_PRS}/mssm_high_ld_regions'):
        compute_high_ld_region(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc',
            f'{PROJECT_DIR}/jeesonja/high_ld_regions_reformatted',
            f'{SCHIZO_DIR_PRS}/mssm_high_ld_regions')

    if not os.path.exists(f'{SCHIZO_DIR_SINGLE_CELL}/schizophrenia_status'):
        extract_phenotype_data(f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata',
                               'IID',
                               'schizophrenia_status',
                               f'{SCHIZO_DIR_SINGLE_CELL}/schizophrenia_status')

    ### LDM COMPUTATION ###
    if not os.path.exists(
            f'{SCHIZO_DIR_PRS}/mclean_ldm.cors.root'):
        compute_ldm_chunks(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc',
            'chunk_list',
            multithread=True, software='megaprs',
            num_variants=512_966, chunk_size=50_000)

        join_ldm_chunks(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc',
            'chunk_list', 'megaprs')

    if not os.path.exists(
            f'{SCHIZO_DIR_PRS}/mssm_ldm.cors.root'):
        compute_ldm_chunks(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc',
            'chunk_list',
            software='megaprs', num_variants=9_513_450, chunk_size=100_000)

        join_ldm_chunks(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc',
            f'{SCHIZO_DIR_PRS}/mssm_ldm',
            'megaprs')

    ### MORE FORMATTING FIXES... ###
    snps = load_dbSNP('hg19')

    # TODO: run this function on the mclean_cohort('s LDM index file).
    get_rs_numbers_bim_or_pvar(f'{SCHIZO_DIR_PRS}/mssm_ldm.cors.bim', snps)

    make_bim_or_pvar_IDs_unique(f'{SCHIZO_DIR_PRS}/mssm_ldm.cors.bim',
                                f'{SCHIZO_DIR_PRS}/mssm_ldm.cors.bim',
                                separator='_')

    make_bim_or_pvar_IDs_unique(f'{SCHIZO_DIR_PRS}/mclean_ldm.cors.bim',
                                f'{SCHIZO_DIR_PRS}/mclean_ldm.cors.bim',
                                separator='_')

    ### SCORE COMPUTATION ###
    if not os.path.exists(f'{SCHIZO_DIR_PRS}/mclean_prs.cors'):
        compute_prs_statistics(
            f'{SCHIZO_DIR_GWAS}/schizo_gwas_summary_stats.megaprs',
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc',
            f'{SCHIZO_DIR_PRS}/mclean_prs', 'megaprs',
            high_ld_regions=None,
            ld_matrix=f'{SCHIZO_DIR_PRS}/mclean_ldm')

    if not os.path.exists(f'{SCHIZO_DIR_PRS}/mssm_prs.cors'):
        compute_prs_statistics(
            f'{SCHIZO_DIR_GWAS}/schizo_gwas_summary_stats.megaprs',
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc',
            f'{SCHIZO_DIR_PRS}/mssm_prs', 'megaprs',
            high_ld_regions=None,
            ld_matrix=f'{SCHIZO_DIR_PRS}/mssm_ldm')

    if not os.path.exists(f'{SCHIZO_DIR_PRS}/mclean_prs.profile'):
        assign_scores(f'{SCHIZO_DIR_PRS}/mclean_prs.effects',
                      f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc',
                      f'{SCHIZO_DIR_PRS}/mclean_prs')
    if not os.path.exists(f'{SCHIZO_DIR_PRS}/mssm_prs.profile'):
        assign_scores(f'{SCHIZO_DIR_PRS}/mssm_prs.effects',
                      f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc',
                      f'{SCHIZO_DIR_PRS}/mssm_prs')
