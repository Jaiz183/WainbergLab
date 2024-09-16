import numpy as np
import polars as pl
from utils import run, get_coding_genes
from single_cell import SingleCell, Pseudobulk, DE
import os, sys
from constants import *
from useful_functions import read_columns, \
    get_principle_components, compute_overlaps, graph_prs_disease_status
import logging
import plotext as pltxt

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# INTERESTING_COLS = {'projid': pl.String,
#                     'study': pl.String,
#                     'nov_seek': pl.String,
#                     'apoe_genotype': pl.String,
#                     'age_first_ad_dx': pl.String,
#                     'niareagansc':
#                         pl.String,
#                     'cogdx': pl.String}

def pseudobulk_expression(rosmap_metadata: pl.LazyFrame,
                          pb_file: str = f'{SINGLE_CELL_DIR}/{PSEUDO_BULK_DIR}',
                          retrieve: bool =
                          False) -> Pseudobulk:
    """
    Compute pseduobulk expression using functions from single_cell.py. Also:
    1) Cleans and loads metadata onto single cell object
    2) Adds PRS scores to single cell object
    3) Adds PCs subset to caucasians to single cell object

    @:param retrieve - if true, retrieve an existing PseudoBulk object from memory. If false, compute pseduobulk expression
    and save it to memory.
    """
    if retrieve:
        return Pseudobulk(pb_file)

    if os.path.exists(f'{SINGLE_CELL_DIR}/{SINGLE_CELL_QC}'):
        sc = (SingleCell(
            f'{SINGLE_CELL_DIR}/{SINGLE_CELL_QC}',
            num_threads=None)
        .with_columns_obs(
            projid=pl.col(f'{PROJ_ID}').cast(pl.String)))
    else:
        sc = (SingleCell(
            f'{PROJECT_DIR}/single-cell/Green/p400_qced_shareable.h5ad',
            num_threads=None)
        .qc(cell_type_confidence_column='cell.type.prob',
            doublet_column='is.doublet.df',
            custom_filter=pl.col.projid.is_not_null()) \
        .with_columns_obs(
            subset=pl.col.subset.replace({'CUX2+': 'Excitatory'}),
            projid=pl.col.projid.cast(pl.String)))

        sc.save(filename=f'{SINGLE_CELL_DIR}/{SINGLE_CELL_QC}',
                overwrite=True)

    # 1. Quality control cell data so that projid is non-null.
    # 2. Modify subset (cell type) and projid columns.
    # Pseudo bulk with appropriate id, cell type columns. Add ROSMAP metadata to covariates with additional obs.
    # Filter by valid genes.
    # Add a couple of columns as covariates before pseudobulking.
    pb = sc.pseudobulk(ID_column=f'{PROJ_ID}',
                       cell_type_column='subset',
                       additional_obs=rosmap_metadata.collect())

    pb = pb.filter_var(pl.col._index.is_in(get_coding_genes()['gene'])) \
        .with_columns_obs(
        apoe4_dosage=pl.col.apoe_genotype.cast(pl.String)
        .str.count_matches('4').fill_null(strategy='mean'),
        pmi=pl.col.pmi.fill_null(strategy='mean'))

    if not retrieve:
        pb.save(pb_file, overwrite=True)

    return pb


def get_rosmap_metadata(metadata_file: str = ROSMAP_METADATA_FILE,
                        id_file: str = ROSMAP_ID_FILE,
                        metadata_ids_file: str = METADATA_IDS_FILE,
                        retrieve: bool = False,
                        custom_filter: list | None = None) -> pl.LazyFrame:
    """
    Returns a lazy frame containing all ROSMAP metadata with corresponding GWAS ID column, PRS scores, and principle components.
    @:param metadata_file - metadata file without IDs.
    @:param id_file - metadata file with IDs.
    @:param retrieve - if True, create updated metadata file and save. Otherwise, load existing metadata file.

    Preconditions:
        - metadata_file and id_file must contain projids

    Post conditions:
        - No rows with null projid or wgs_id
    """

    if retrieve:
        return pl.scan_csv(metadata_ids_file)

    # Get metadata and join with GWAS ID to assign race data + PRSs.
    metadata = read_columns(metadata_file, data_types={PROJ_ID: str}).unique(
        subset=f'{PROJ_ID}')

    # GWAS IDs.
    gwas_ids = (
        read_columns(id_file, [WGS_ID, PROJ_ID], data_types={PROJ_ID: str})
        .unique(subset=f'{PROJ_ID}'))

    metadata_ids = (metadata.join(gwas_ids, on=PROJ_ID, how='left')
                    .drop_nulls(subset=[PROJ_ID, WGS_ID]))

    # Add PRS scores. Normalise if not already normalised.
    if not os.path.exists(f'{SCORES_DIR}/{SCORES_FILE_NORMALISED}'):
        normalise_prs_scores(f'{SCORES_DIR}/{SCORES_FILE}')

    rosmap_meta_prs = add_prs_scores(metadata_ids,
                                     f'{SCORES_DIR}/{SCORES_FILE_NORMALISED}')

    if not os.path.exists(f'{SINGLE_CELL_DIR}/{PC_FILE}.eigenvec'):
        get_principle_components(RACE_FILTERED_DATA, 10, PC_FILE)
    # Add principle components.
    rosmap_meta = add_principle_components_to_pseudobulk(rosmap_meta_prs,
                                                         f'{PC_FILE}.eigenvec')

    logger.debug(
        f'Number of entries removed after adding PRS scores: {metadata_ids
        .collect().item()} '
        f'- {rosmap_meta_prs.select(pl.len()).collect().item()} = '
        f'{metadata_ids.select(pl.len()).collect().item() - rosmap_meta_prs.select(pl.len()).collect().item()}')

    # Create new cognitive function definition.
    rosmap_meta = rosmap_meta.with_columns(
        dx_cont=pl.when(pl.col.cogdx == 1).then(0)
        .when(pl.col.cogdx.is_in([2, 3])).then(1)
        .when(pl.col.cogdx.is_in([4, 5])).then(2)
        .otherwise(None))

    # Filter.
    if custom_filter is not None:
        for flt in custom_filter:
            rosmap_meta = flt(rosmap_meta)

    rosmap_meta.collect().write_csv(metadata_ids_file)

    return rosmap_meta


def differential_expression(pb: Pseudobulk = None,
                            independent_variables: list[str] = None,
                            covariates: list[str] = None,
                            retrieve: bool = False,
                            filter_non_diseased: bool = False) -> \
        list[DE]:
    """
    Performs differential expression. Code from Keon.

    :param covariates - covariates for regression fit
    :param independent variables - variable that is varied
    :param filter_non_diseased - if True, only dx_cont scores of 0 are kept, else everything is kept.


    Pre-conditions:
     - If retrieve is True, only independent variables must be specified, else everything is specified.
    """
    # Change label_column to prs_scores if necessary. Try NIA-REAGAN?
    differential_expressions = []

    if retrieve:
        logger.info('Retrieving existing DEs...')
        for independent_variable in independent_variables:
            differential_expressions.append(
                DE(f'{DE_DIR}/{independent_variable}_{DE_FILE}'))

        return differential_expressions

    for independent_variable in independent_variables:
        if independent_variable == SCORE_COL and filter_non_diseased:
            # Only keep non-diseased samples for score DE if regressing PRS score.
            logger.info('Removing diseased samples...')
            pb = pb.filter_obs(pl.col('dx_cont').eq(0))

        de = pb.qc(case_control_column=None,
                   custom_filter=pl.col.dx_cont.is_not_null()) \
            .DE(label_column=independent_variable,
                case_control=False,
                covariate_columns=covariates,
                include_library_size_as_covariate=True)
        de.plot_voom(
            save_to=f'{DE_DIR}/{independent_variable}_{DE_PLOT}',
            overwrite=True)

        logger.info(f'Saving DE for {independent_variable}...')
        de.save(f'{DE_DIR}/{independent_variable}_{DE_FILE}', overwrite=True)
        differential_expressions.append(de)

    return differential_expressions


def add_principle_components_to_pseudobulk(metadata: pl.LazyFrame,
                                           pc_file: str) -> pl.LazyFrame:
    """
    Adds PC scores to metadata file. Since PCs are only computed for a subset of the population (say caucasians),
    inner strategy subsets to that strata.

    Preconditions:
        - PC columns must be named PC[n]
    """
    pc_columns = [f'PC{num}' for num in range(1, 11)]
    principle_components = (read_columns(pc_file,
                                         None,
                                         header=False,
                                         column_names=[FAMILY_ID,
                                                       INDIVIDUAL_ID] + pc_columns,
                                         delimiter=' ')
                            .drop(FAMILY_ID))
    # Add PCs as sample-level covariates.
    metadata = (metadata
                .join(principle_components, how='inner', left_on=WGS_ID,
                      right_on=INDIVIDUAL_ID))
    # logger.debug(metadata.collect())
    return metadata


def normalise_prs_scores(prs_file: str):
    """
    Writes normalised PRS scores to SCORES_FILE_NORMALISED.
    """

    prs_scores = read_columns(prs_file, None,
                              delimiter='\t',
                              data_types={INDIVIDUAL_ID: str, FAMILY_ID: str})
    # Standardise by computing z-scores.
    scores = prs_scores.select(SCORE_COL)
    mean = scores.mean().collect().item()
    std = scores.std().collect().item()
    prs_scores = prs_scores.with_columns(
        SCORE1_AVG=pl.col(SCORE_COL).sub(mean).truediv(std))

    prs_scores.collect().write_csv(f'{SCORES_DIR}/{SCORES_FILE_NORMALISED}',
                                   separator='\t')


def add_prs_scores(metadata: pl.LazyFrame, prs_file: str) -> pl.LazyFrame:
    """
    Return metadata with prs scores added.
    """
    prs_scores = read_columns(prs_file, [INDIVIDUAL_ID, SCORE_COL],
                              delimiter='\t', data_types={INDIVIDUAL_ID: str})
    return (metadata.join(prs_scores, how='left', left_on=WGS_ID,
                          right_on=INDIVIDUAL_ID)
            .drop_nulls(subset=SCORE_COL))


def filter_by_race(race: str,
                   metadata_file: str = METADATA_IDS_FILE,
                   filtered_file: str = RACE_FILTERED_DATA):
    """
    Filters quality controlled, scored target data by race and writes to filtered_file.

    Preconditions:
        - metadata_file must have already been joined with GWAS IDs from an ID file.

    :param filtered_file - name of quality controlled BED file filtered by race given.
    :param metadata_file - name of file with ROSMAP metadata.
    """

    race_data = (read_columns(metadata_file, [WGS_ID, RACE_COL])
                 .drop_nulls(subset=WGS_ID))
    if race not in RACES:
        raise ValueError(f"That isn't a real race...")
    else:
        race_code = RACES[race]

    # logger.debug(race_data.collect())

    logger.info(f'Filtering metadata file ({metadata_file})...')
    race_filtered_data = race_data.filter(pl.col(RACE_COL).eq(race_code))
    logger.info(
        f'Number of entries removed after filtering for {race} race: {race_data.select(pl.len()).collect().item()}'
        f' - {race_filtered_data.select(pl.len()).collect().item()} = {race_data.select(pl.len()).collect().item()
                                                                       - race_filtered_data.select(pl.len()).collect().item()}')
    # Now, duplicate gwas_id to IID column and rename existing gwas_id column to FID column. Also, drop race column.
    race_filtered_data = (race_filtered_data
                          .with_columns(IID=pl.col(f'{WGS_ID}'))
                          .rename({WGS_ID: FAMILY_ID})
                          .drop(f'{RACE_COL}'))

    logger.info(f'Saving filtered file as {filtered_file}...')
    # Write to file!
    race_filtered_data.collect().write_csv(CAUCASIAN_SAMPLES, separator='\t')

    logger.info(f'Filtering binary file ({QC_TARGET_DATA_WITHOUT_SCORES})...')
    run(f'{PLINK2} '
        f'--bfile {QC_TARGET_DATA_WITHOUT_SCORES} '
        f'--keep {CAUCASIAN_SAMPLES} '
        f'--make-bed '
        f'--out {filtered_file}')


def filter_cognitively_diseased(rosmap_metadata: pl.LazyFrame) -> pl.LazyFrame:
    """
    Keeps only those samples with zero dx_cont scores (non-cognitively diseased).
    """
    return rosmap_metadata.filter(pl.col('dx_cont').eq(0))


def graph_fdr(differential_expression: DE):
    """
    Graphs num hits obtained against FDR threshold.
    """
    fdrs = np.arange(0.05, 0.3, 0.001)
    num_hits = []

    for fdr in fdrs:
        hits: pl.DataFrame = differential_expression.get_num_hits(threshold=fdr)
        aggregate_hits = hits.select("num_hits").sum().item()
        num_hits.append(aggregate_hits)
        logger.debug(aggregate_hits)

    pltxt.scatter(fdrs, num_hits, marker='x')
    pltxt.yscale('linear')
    pltxt.title('FDR vs hits')
    pltxt.show()


if __name__ == '__main__':
    # Compute PCs.
    # filter_by_race('caucasian')
    # get_principle_components(RACE_FILTERED_DATA, 10, f'{PC_FILE}')

    rosmap_meta = get_rosmap_metadata(retrieve=True)
    # Compute pseudobulk expression separately for non-diseased and diseased.
    pb_all = pseudobulk_expression(rosmap_meta, retrieve=True)

    # Perform differential expression.
    independent_variables = ['dx_cont', SCORE_COL]
    covariates_prs = ['age_death', 'sex', 'pmi', 'apoe4_dosage', 'dx_cont'] + [
        f'PC{num}' for num in range(1, 11)]
    covariates_dx_cont = ['age_death', 'sex', 'pmi', 'apoe4_dosage']

    # Compute differential expression for prs, then dx_cont.
    # differential_expressions = (differential_expression(pb_all,
    #                                                     [independent_variables[
    #                                                          1]],
    #                                                     covariates_prs,
    #                                                     retrieve=False,
    #                                                     filter_non_diseased=False) +
    #                             differential_expression(
    #                                 pb_all,
    #                                 [independent_variables[0]],
    #                                 covariates_dx_cont,
    #                                 filter_non_diseased=False, retrieve=False))
    #
    differential_expressions = differential_expression(retrieve=True,
                                                       independent_variables=independent_variables)

    all_hits = []
    for i in range(len(differential_expressions)):
        differential_expression = differential_expressions[i]
        hits: pl.DataFrame = differential_expression.get_hits(
            threshold=0.1)
        all_hits.append(hits)

        print(f'{differential_expression}:\n'
              f'Hits:\n {hits}')

    # Plot dx_cont, PRS relationship.
    rosmap_meta_df = rosmap_meta.collect()
    graph_prs_disease_status(rosmap_meta_df, 'dx_cont', SCORE_COL,
                             save_to=f'{
                             SINGLE_CELL_DIR}/dx_cont_PRS_swarm.pdf',
                             plot_type='swarm')
    graph_prs_disease_status(rosmap_meta_df, 'dx_cont', SCORE_COL,
                             save_to=f'{
                             SINGLE_CELL_DIR}/dx_cont_PRS_box.pdf',
                             plot_type='box')

    # Find overlaps.
    print(compute_overlaps(all_hits))

    # Sanity check.
    print(all_hits[0].filter(pl.col('gene').eq('TLR3')))
