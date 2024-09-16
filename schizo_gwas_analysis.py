import subprocess

import polars as pl

from utils import (run, merge_bfiles, load_dbSNP,
                   get_rs_numbers_bim_or_pvar, make_bim_or_pvar_IDs_unique,)
from constants import *
from useful_functions import read_columns, \
    remove_missing_variant_ids_from_genotypes, remove_heterozygosity, update_sex
import logging
import sys, os

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# TODO: doesn't work as is because of formatting issues with VCF input.
def filter_vcf(vcf_files: dict[str, str]) -> \
        None:
    """Filters VCF files and converts to pfiles."""
    for vcf_file, save_to in vcf_files.items():
        try:
            run(f'{PLINK2} '
                f'--vcf {vcf_file} '
                f'--vcf-min-gq 20 '
                f'--vcf-min-dp 10 '
                f'--vcf-min-qual 30 '
                f'--double-id '
                f'--make-pgen '
                f'--out {save_to}')
        except subprocess.CalledProcessError:
            logger.error(f'Can\'t filter the vcf file: {vcf_file}!')

# TODO: might want to change this to make-pgen once a PRS scoring method
#  compatible with pfiles becomes available. Could convert to .gen files.
def quality_control(genotype_file: str, save_to: str) -> None:
    """
    Quality controls GWAS data.

    :param genotype_file: file to perform quality control on
    :param save_to: where to save quality controlled output to
    """
    # Heterozygosity control.
    remove_heterozygosity(genotype_file, 'genotypes_no_hetero')

    run(f'rm {genotype_file}.*')

    logger.info('Basic quality control - MAF, HWE, missing / incorrect data...')
    run(f'{PLINK2} '
        f'--bfile genotypes_no_hetero '
        f'--make-bed '
        f'--double-id '
        f'--max-maf 0.99 '
        f'--maf 0.01 '
        f'--geno '
        f'--hwe 1e-15 midp '
        f'--mind '
        f'--autosome '
        f'--out {save_to}')

    run(f'rm genotypes_no_hetero.*')

    # Uncomment to threshold by LD.
    # LD pruning.
    # remove_linkage_disequilibrium('genotypes_no_hetero', window_size=50,
    #                               correlation_coefficient=0.8,
    #                               genome_file_pruned=save_to)


def standardise_metadata_ids(metadata_file: str, save_to: str) -> None:
    """
    Used to keep CMC IDs where available and CC IDs where not available.

    Pre-conditions:
        - Must be run after correct_metadata_ids
    """
    metadata = read_columns(metadata_file,
                            delimiter='\t')
    # IID is ID when CMC ID is null and CMC ID otherwise.
    metadata = metadata.with_columns(
        IID=pl.when(pl.col('CMC_ID').eq('CMC_NA')).then(pl.col('ID')).otherwise(
            pl.col('CMC_ID')))

    metadata.rename({'IID': '#IID', 'Gender': 'SEX'}).drop('CMC_ID',
                                                           'ID').collect().write_csv(
        save_to, separator='\t')


def correct_cmc_ids(metadata_file: str) -> None:
    """
    Adds CMC_ prefix to all CMC IDs to match formatting of genotype metadata.
    """
    metadata = read_columns(metadata_file, delimiter='\t')
    metadata = metadata.with_columns(CMC_ID='CMC_' + pl.col('CMC_ID'))
    metadata.collect().write_csv(
        f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata_corrected',
        separator='\t')


def create_sex_annotations(metadata_file: str, save_to: str) -> None:
    """Creates sex annotation file."""
    metadata = read_columns(metadata_file, delimiter='\t',
                            columns=['#IID', 'SEX']).select(['#IID', 'SEX'])
    metadata.collect().write_csv(save_to, separator='\t')


def pfile_as_bcf(genotype_file: str, save_to: str, pfile_to_bcf: bool) -> None:
    """
    Exports pfile to BCF format or vice versa.
    :param pfile_to_bcf: True if converting from pfile to BCF else False.
    """
    if pfile_to_bcf:
        run(f'{PLINK2} '
            f'--pfile {genotype_file} '
            f'--export bcf '
            f'--out {save_to}')
    else:
        run(f'{PLINK2} '
            f'--bcf {genotype_file} '
            f'--make-pgen '
            f'--out {save_to}')


def pfile_as_gen(genotype_file: str, save_to: str, pfile_to_gen: bool) -> None:
    """
    Exports pfile to .gen format or vice versa.
    :param pfile_to_gen: True if converting from pfile to .gen else False.
    """
    if pfile_to_gen:
        run(f'{PLINK2} '
            f'--pfile {genotype_file} '
            f'--export oxford-v2 '
            f'--out {save_to}')

        # Compress.
        run(f'gzip {save_to}.gen')
    else:
        run(f'{PLINK2} '
            f'--gen {genotype_file} '
            f'--make-pgen '
            f'--out {save_to}')


if __name__ == '__main__':
    # Filter VCF files.
    vcf_files = {
        f'{SCHIZO_DIR_CMC_GENOTYPES}/CMC_OmniExpressExome_ImputationHRC_chr{i}.dose.vcf.gz':
            (f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_chr{i}')
        for i in
        range(1, 23)
    }

    correct_cmc_ids(
        f'{SCHIZO_DIR_SINGLE_CELL_RAW}/individual_metadata_filtered_grouped.tsv')

    standardise_metadata_ids(
        f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata_corrected',
        f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata_corrected')

    create_sex_annotations(
        f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata_corrected',
        f'{SCHIZO_DIR_SINGLE_CELL}/sex_annotations')

    vcf_files[
        f'{SCHIZO_DIR_SINGLE_CELL_RAW}/SZBDMulti-Seq_Cohort_Genotypes.vcf.gz'] = \
        f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes'

    input_output = vcf_files.items()

    for vcf_file, save_to in input_output:
        if not os.path.exists(f'{save_to}.bed'):
            update_sex(
                f'{vcf_file}',
                f'{SCHIZO_DIR_SINGLE_CELL}/sex_annotations',
                f'{save_to}')

            # quality control

    if not os.path.exists(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes.bed'):
        merge_bfiles(list(vcf_files.values())[:-1],
                     f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes')

    genome_build = load_dbSNP('hg19')
    if not os.path.exists(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc.bed'):
        quality_control(f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes',
                        f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc')
        get_rs_numbers_bim_or_pvar(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc.bim',
            genome_build)
        make_bim_or_pvar_IDs_unique(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc.bim',
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc.bim',
            separator='_')
        # Uncomment this if you want to LD prune.
        # remove_missing_variant_ids_from_genotypes(
        #     f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc',
        #     f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc')

    if not os.path.exists(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc.bed'):
        quality_control(f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes',
                        f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc')
        get_rs_numbers_bim_or_pvar(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc.bim',
            genome_build)
        make_bim_or_pvar_IDs_unique(
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc.bim',
            f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc.bim',
            separator='_')
        # Uncomment this if you want to LD prune.
        # remove_missing_variant_ids_from_genotypes(
        #     f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc',
        #     f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc')
