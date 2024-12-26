from multiprocessing import Pool

import polars as pl
from utils import run
import os, sys
from constants import *
import logging
import seaborn as sns
import numpy as np
import random

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def random_sample(cells: np.ndarray, pt_kept: float, retrieve: bool,
                  save_to: str | None) -> tuple[np.ndarray, list[int]]:
    """
    Randomly samples pt_kept% of cells. Saves if retrieve is false and save_to
    is not None, else loads from save_to. save_to cannot be None and retrieve
    cannot be True simultaneously.
    """
    if retrieve:
        return np.load(f'{save_to}.npy'), np.load(f'{save_to}_indices.npy')

    sample_indices = random.sample(list(range(len(cells))),
                                   int((pt_kept / 100) * len(cells)))
    sample_filter = [False] * len(cells)
    for index in sample_indices:
        sample_filter[index] = True

    sample = cells[sample_filter]

    if save_to is not None:
        np.save(f'{save_to}.npy', sample)
        np.save(f'{save_to}_indices.npy', sample_indices)
    return sample, sample_indices

def split_data(pcs: np.ndarray, proportion: float, save_train_to: str | None,
               save_test_to: str | None) -> tuple[np.ndarray, np.ndarray]:
    """
    :param proportion: percentage to be used as training data.
    :param save_train_to: saves training data to the location specified iff not None.
    :param save_test_to: saves test data to the location specified iff not None.
    """
    if not os.path.exists(f'{KNN_DIR}/rosmap_pcs_train_full.npy'):
        train_data, train_data_indices = random_sample(pcs,
                                                       proportion,
                                                       False,
                                                       save_train_to)
        # Make remaining test data by just removing any vector in train_data.
        test_filter = [True] * len(pcs)
        for train_data_index in train_data_indices:
            test_filter[train_data_index] = False

        test_data = pcs[test_filter]

        if save_train_to is not None:
            np.save(save_train_to, train_data)

        if save_test_to is not None:
            np.save(save_test_to, test_data)

    else:
        train_data = np.load(f'{save_train_to}.npy')
        test_data = np.load(f'{save_test_to}.npy')

    return train_data, test_data

def read_columns(lf: str, columns: list[str] = None, delimiter: str = ',',
                 header: bool = True, data_types: dict[str, type] | None = None,
                 column_names: list[str] | None = None) -> (
        pl.LazyFrame):
    """
    Reads a file with polars. Drops null rows, columns not in columns parameter.
    :param columns - if None, read all columns, else read columns specified.
    :param header - reads header line if True, else doesn't.
    :param data_types - specify data type of entries in column as
    column_name: data_type key-value pairs.
    """
    if data_types is None:
        data_types = {}
    lf = pl.scan_csv(
        f'{lf}', separator=delimiter, has_header=header,
        truncate_ragged_lines=True, dtypes=data_types,
        new_columns=column_names)

    return (lf.drop(col for col in lf.columns if col not in
                    columns)) if columns is not None else lf


def pfile_bfile(input_file: str, output_file: str,
                pfile_to_bfile: bool) -> None:
    """
    Converts between pfiles and bfiles.
    :param pfile_to_bfile: converts from pfile to bfile if True,
    else converts from bfile to pfile.
    """
    if pfile_to_bfile:
        run(f'{PLINK2} '
            f'--pfile {input_file} '
            f'--make-bed '
            f'--out {output_file}')
    else:
        run(f'{PLINK2} '
            f'--bfile {input_file} '
            f'--make-pgen '
            f'--out {output_file}')


def merge_bcf_files(input_files: list[str], save_to: str,
                    duplicates: bool) -> None:
    """
    Merges BCF files into one file.
    :param duplicates: True if duplicate entries exist else False.
    """
    run(f'{BCF} '
        f'merge '
        f'-o {save_to} '
        f'{'--force-samples ' if duplicates else ''}'
        + ' '.join(input_files))


def remove_linkage_disequilibrium(genome_file_pruned: str, genome_file: str,
                                  window_size: int = 100,
                                  correlation_coefficient: float = 0.8,
                                  step: int = 1, ) -> None:
    """
    Writes LD pruned genome_file to genome_file_pruned. Temporarily creates
    .in, .out files (variants to keep, exclude), but are cleaned up promptly.
    :param genome_file: bfile to remove linkage disequilibrium from (contains genotypes)
    :param window_size: size of window over which linkage disequilibrium is
    eliminated on every iteration in kilobases. Larger window sizes are
    stricter.
    :param correlation_coefficient: how strong the correlation between two
    alleles of a haplotype must be to reject them. Lower coefficients are
    stricter.
    """
    run(f'{PLINK2} '
        f'--bfile {genome_file} '
        f'--indep-pairwise {window_size}kb {step} {correlation_coefficient} '
        f'--out {genome_file_pruned}')

    run(f'{PLINK2} '
        f'--bfile {genome_file} '
        f'--extract {genome_file_pruned}.prune.in '
        f'--make-bed '
        f'--out {genome_file_pruned}')

    run(f'rm {genome_file_pruned}.prune.in {genome_file_pruned}.prune.out')


def get_principle_components(genome_file: str, num_pcs: int, save_to: str):
    """
    Gets a specified number of principal components from a given genome file.
    :param genome_file - file to compute PCs from.
    :param save_to - file to write principal components to.
    """
    run(f'{GCTA} '
        f'--bfile {genome_file} '
        f'--make-grm '
        f'--out {genome_file}_grm')

    run(f'{GCTA} '
        f'--grm {genome_file}_grm '
        f'--pca {num_pcs} '
        f'--out {save_to}')

    (read_columns(f'{save_to}.eigenvec', delimiter=' ',
                  column_names=['FID', 'IID'] + [f'PC{num}'
                                                 for num in range(1, num_pcs +
                                                                  1)]).collect()
     .write_csv(f'{save_to}.eigenvec'))

    run(f'rm -f {genome_file}_grm.*')


def remove_missing_variant_ids_from_genotypes(
        genotype_file: str = RACE_FILTERED_DATA,
        save_to: str = RACE_FILTERED_DATA_NO_MISSING_IDS):
    """
    Removes entries with missing variant IDs from genome file, writing to the
    specified output file.
    :param genotype_file: file to remove missing variant IDs from.

    """
    remove_missing_variant_ids_from_indices(f'{genotype_file}.bim',
                                            f'missing_variants')

    # Remove missing variant ID entries.
    run(f'{PLINK2} '
        f'--bfile {genotype_file} '
        f'--extract missing_variants '
        f'--make-bed '
        f'--out {save_to}')

    # Clean up temporary file.
    run(f'rm missing_variants')


def remove_missing_variant_ids_from_indices(index_file: str, save_to: str):
    """
    Removes missing variant ID entries from genome file's index file, writing to
    specified output file. Reports number of lines (entries) lost.
    """

    # Keep entries that aren't missing variant IDs, cut out only variant IDs.
    condition = '\'$2 != "."\''

    logger.info(f'Started with: ')
    run(f'wc -l {index_file}')
    logger.info(f' lines.')

    run(f'{AWK} '
        f'{condition} '
        f'{index_file} '
        f'| cut -f 2 '
        f'> {save_to}')

    logger.info(f'Ended with: ')
    run(f'wc -l {save_to}')
    logger.info(f' lines.')


def add_principal_components_to_metadata(metadata: pl.LazyFrame, pc_file: str,
                                         num_pcs: int, metadata_id: str,
                                         pc_id: str) -> pl.LazyFrame:
    """
    Adds principal components computed from get_principle_components to a
    metadata file.
    :param metadata: LazyFrame corresponding to metadata.
    :param pc_file: file with principal component scores for each variant.
    :param num_pcs: number of principal components.
    :param metadata_id: ID column in metadata file shared with pc_file.
    :param pc_id: ID column in pc_file shared with metadata_file.
    :return: LazyFrame that contains the augmented metadata.

    Preconditions:
        - pc_file ust have headings in the PC\d format.
    """
    pc_cols = [f'PC{num}' for num in range(1, num_pcs + 1)]
    pc = read_columns(pc_file, delimiter='\t')
    metadata_with_pcs = (metadata.join(left_on=metadata_id, right_on=pc_id,
                                       how='left',
                                       other=pc).drop_nulls(subset=pc_cols))
    return metadata_with_pcs


def report_lost_entries(prev_metadata: pl.DataFrame,
                        new_metadata: pl.DataFrame) -> None:
    """
    Debugs the number of entries lost when filtering from prev to new_metadata.
    """
    logger.info(
        f'Lost {prev_metadata.select(pl.len()).item()} '
        f'- {new_metadata.select(pl.len()).item()} = {prev_metadata.select(pl.len()).item() - new_metadata.select(pl.len()).item()} entries.')


def compute_overlaps(hits: list[pl.DataFrame]) -> pl.DataFrame:
    """
    Computes overlap (genes in all entries of hits) between gene hits.
    :param hits: hits from different analyses. Must have a gene column that
    lists genes.
    """
    final = hits[0]
    for i in range(1, len(hits)):
        final = final.join(hits[i], how='inner', on='gene', suffix=f'{i}')

    return final


def graph_prs_disease_status(metadata: pl.DataFrame, prs_col: str,
                             disease_stat_col: str, plot_type: str,
                             save_to: str,
                             additional_feature: str = None) -> None:
    """
    Graphs PRS against disease status to visualise correlation between the 
    two variables.
    
    :param plot_type: type of plot. Must be one of swarm or box.
    :param save_to: location to save plot to.
    :param additional_feature: if required, adds hue to represent this
    feature.
    """
    if plot_type == 'swarm':
        plot = sns.swarmplot(metadata, x=disease_stat_col, y=prs_col,
                             hue=additional_feature)

    elif plot_type == 'box':
        plot = sns.boxplot(metadata, x=disease_stat_col, y=prs_col,
                           hue=additional_feature)
    else:
        raise ValueError(f'The plot type {plot_type} is not supported.')

    fig = plot.get_figure()
    fig.savefig(save_to)


def remove_heterozygosity(genotype_file: str, save_to: str) -> None:
    """
    Reduces heterozygosity in a genotype file by removing entries with
    abnormal (beyond 3 std deviations) F coefficients. Makes temporary files,
    but cleans them up immediately.
    """
    run(f'{PLINK2} '
        f'--bfile {genotype_file} '
        f'--het '
        f'--keep {genotype_file}.fam '
        f'--out genotypes_for_hetero')

    # Remove individuals with F coefficients that are more than 3 standard.
    # deviation (SD) units from the mean.

    data = pl.scan_csv(f'genotypes_for_hetero.het', separator='\t',
                       dtypes={'#FID': str, 'IID': str})
    f_coeff = data.select("F")
    mean = f_coeff.mean().collect().item()
    std = f_coeff.std().collect().item()

    upper_bd = mean + 3 * std
    lower_bd = mean - 3 * std

    valid = data.filter(pl.col("F") <= upper_bd, pl.col("F") >= lower_bd)
    valid.collect().write_csv(
        f'valid_hetero',
        separator="\t",
        include_header=True)

    # Remove all anomalous heterozygosity entries from main file.
    run(f'{PLINK2} '
        f'--bfile {genotype_file} '
        f'--keep valid_hetero '
        f'--make-bed '
        f'--out {save_to} ')

    # Clean up unnecessary files.
    run(f'rm -f valid_hetero genotypes_for_hetero')


def update_sex(genotype_file: str, sex_annotations: str, save_to: str) -> None:
    """
    Updates sample sexes if given metadata.
    :param sex_annotations: sample sexes. Sex column must be third or
    labelled 'SEX', preceded by FID and IID columns. Sexes must follow ^[m |
    M | f | F]* or ^[1 | 0]* convention.
    """
    run(f'{PLINK2} '
        f'--vcf {genotype_file} '
        f'--update-sex {sex_annotations} '
        f'--double-id '
        f'--make-bed '
        f'--split-par hg38 '
        f'--out {save_to}')


def compute_ldm_chunk(genome_file: str, chunk_list: str, software: str, *args):
    """
    Computes the LDM for a single chunk of the genome / chromosome file.
    :param chunk_list: file to which current LDM chunk file's name is
    written. Used in join_ldm_chunks.
    :param genome_file: full genome file.
    :param software: must be one of sbayesr or megaprs.
    :param args:
        - If using sbayesr, args must contain start and end genomic locations of
        chunk desired in that order (chunks by genomic section).
        - If using megaprs, args must contain chromosome number (chunks by
        chromosome).
    """
    match software:
        case 'sbayesr':
            start, end = args
            run(f'{GCTB} '
                f'--bfile {genome_file} '
                f'--make-sparse-ldm '
                f'--snp {start}-{end} '
                f'--out {genome_file}')
            run(f'echo {genome_file}.snp{start}-{end}.ldm.full >> {chunk_list}')
        case 'megaprs':
            chrom = args[0]
            run(f'{LDAK} '
                f'--bfile {genome_file} '
                f'--calc-cors {genome_file}_chr{chrom} '
                f'--chr {chrom}')
            run(f'echo {genome_file}_chr{chrom} >> {chunk_list}')
        case _:
            raise ValueError(f"Sorry, I haven't written code for {software} "
                             f"software yet!")


def compute_ldm_chunks(genome_file: str, chunk_list: str, software: str,
                       multithread: bool = True, **args) -> None:
    """
    Computes sparse LD matrix for chromosomes individually and breaks up
    computation to save time and space by multithreading.
    :param chunk_list: file to which all LDM chunk files' names are
    written.
    :param args:
        - chunk_size - sizes of portions into which genome file is split for
        - num_variants - number of variants in genome file
    multithreading.

    Pre-conditions:
        - num_variants is not None and chunk_size is not None if software == sbayesr
        - 20 GiB of memory per core required
    """
    tasks = []
    # Clear chunk list.
    run(f'echo "" > {chunk_list}')
    if software == 'sbayesr':
        chunk_size = args['chunk_size']
        num_variants = args['num_variants']

        tasks = [(genome_file, software, i, min(i + chunk_size - 1,
                                                num_variants)) for i in
                 range(1, num_variants, chunk_size)]
    elif software == 'megaprs':
        tasks = [(genome_file, chunk_list, software, chrom) for chrom in
                 range(1,
                       23) if
                 not os.path.exists(f'{genome_file}_chr{chrom}.cors.bin')]

    if multithread:
        results = []
        p = Pool(processes=os.cpu_count())
        with p:
            for task in tasks:
                results.append(p.apply_async(compute_ldm_chunk, task))

            for result in results:
                result.wait()

        p.join()
    else:
        for task in tasks:
            compute_ldm_chunk(*task)


def join_ldm_chunks(save_to: str, chunk_list_name: str, software: str) -> None:
    """
    Joins chunked LDMs into one file.
    :param chunk_list_name: file that lists LDM chunk files. Produced by
    compute_ldm_chunks.

    Pre-conditions:
        - Must be run after compute_ldm_chunks, which produces the chunk list required.
    """
    match software:
        case 'sbayesr':
            run(f'{GCTB} '
                f'--mldm {chunk_list_name} '
                f'--make-sparse-ldm '
                f'--out {save_to}')

        case 'megaprs':
            run(f'{LDAK} '
                f'--join-cors {save_to} '
                f'--corslist {chunk_list_name} ')

        case _:
            return

def compute_ldm(genome_file: str) -> None:
    """
    Computes LDM all at once. Don't use this because it is VERY slow.
    """
    run(f'{GCTB} '
        f'--bfile {genome_file} '
        f'--make-sparse-ldm '
        f'--out {genome_file}')


def compute_prs_statistics(summary_statistics_file: str,
                           genome_file: str, save_to: str, software: str,
                           **args):
    """
    Compute PRS scores using summary statistics for a given genome file.

    :param software - method by which PRSs are calculated. Choose one of
    sbayesr or megaprs temporarily.
    :param args: additonal arguments as required by the software chosen.
        - megaprs requires an ld_matrix argument specifying where the LDM is
        stored and, optionally, a high_ld_regions argument specifying regions of
        high LD in the genome build used by the genome file.

    Pre-conditions:
        - Run compute_high_ld_region if you want to specify the high LD regions.
        - Run compute_ldm_chunks to compute LDM.
    """
    match software:
        case 'sbayesr':
            run(f'{GCTB} '
                f'--sbayes R '
                f'--ldm {genome_file}.ldm.sparse '
                f'--pi 0.95,0.02,0.02,0.01 '
                f'--gamma 0.0,0.01,0.1,1 '
                f'--gwas-summary {summary_statistics_file} '
                f'--chain-length 10000 '
                f'--burn-in 2000 '
                f'--out-freq 10 '
                f'--out {save_to}')

        case 'megaprs':

            run(f'{LDAK} '
                f'--mega-prs {save_to} '
                f'--model elastic '
                f'--summary {summary_statistics_file} '
                f'--power -0.25 '
                f'--cors {args['ld_matrix']} '
                f'{f'--high-LD {args['high_ld_regions']}/genes.predictors.used '
                if args['high_ld_regions'] is not None else '--check-high-LD NO '}'
                f'--allow-ambiguous YES '
                f'--extract {summary_statistics_file}')


def find_empty_rows(file: str) -> None:
    """
    Finds empty rows in a file and prints line numbers.
    """
    run(f'awk \'$1 != \'\s\'{{print NR}}\' {file}')
