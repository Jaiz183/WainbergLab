"""
If this module doesn't run well, particularly if the process corresponding to
this module is repeatedly killed, try requesting more memory. Methods like
load_single_cell_data and load_dbSNP require around 100-200 GiB of memory.
"""


import polars as pl
import scanpy
import scipy

import os, sys
from constants import *
import seaborn as sns
import matplotlib as plt

from utils import debug
from single_cell import SingleCell, Pseudobulk, DE
from useful_functions import (read_columns, get_principle_components,
                              add_principal_components_to_metadata,
                              compute_overlaps,
                              graph_prs_disease_status)
import logging
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

debug()


def load_single_cell_data(file: str, retrieve: bool, save_location: str |
                                                                    None,
                          **kwargs) -> (
        SingleCell):
    """
    Loads and quality controls single-cell data. Reads raw counts from
    AnnData object and gene names from RDS file. QC based on mitochondrial
    count, minimum genes read from cell. Creates two classifications - based
    on broad cell types and cell subtypes. In subtype classification, Ex-NRGN is
    ignored because of a lack of sufficient samples, fewer than the number of
    covariates, which makes estimating betas for the covariates during DE
    indeterminate.

    :param file: path to single-cell data
    :param retrieve: retrieves existing, quality controlled single-cell data
    from save_to if true, else loads raw data, QCs and saves to save_to
    :param save_location: where to save quality controlled data

    Preconditions:
        - file refers to an h5ad file.
        - If retrieve is true, save_location must exist
    """
    if retrieve:
        sc = SingleCell(save_location, num_threads=None)
    else:
        sc = SingleCell(file, num_threads=None)

        # Retrieve raw counts.
        sc_anndata = scanpy.read_h5ad(file)
        sc.X = scipy.sparse.csr_array(sc_anndata.layers['counts'])

        # Retrieve gene names.
        # Temporarily commented because of rpy2, ryp R interpreter
        # initialisation issue. If you want to load up a fresh single cell
        # object, uncomment.

        # import rpy2.robjects as ro
        # from rpy2.robjects.packages import importr
        # from rpy2.robjects import pandas2ri
        # base = importr('base')
        # readRDS = base.readRDS
        # var_r = readRDS(f'{SCHIZO_DIR_SINGLE_CELL}/r_components/var')
        # with (ro.default_converter + pandas2ri.converter).context():
        #     var = ro.conversion.get_conversion().rpy2py(var_r)
        #     var = pd.DataFrame(var)
        #     var = pl.DataFrame(var)
        #     var.columns = ['gene']
        #
        # sc.var = var
        # sc = sc.set_var_names('gene')

        # Perform quality control, reject null ID entries, and create a
        # broad cell type classification for better power option later.
        sc = (sc
              .qc(None, None, max_mito_fraction=0.1, min_genes=200,
                  allow_float=True,
                  custom_filter=pl.col('ID').is_not_null().and_(pl.col(
                      'strict_cell_type').ne(
                      'Ex-NRGN')))
              .with_columns_obs(pl.col('Celltype')
                                .cast(pl.String)
                                .str
                                .split_exact('-', 1)
                                .struct.field('field_0')
                                .replace({
            'Ast': 'Astrocyte',
            'Endo': 'Endothelial',
            'Ex': 'Excitatory',
            'In': 'Inhibitory',
            'Mic': 'Microglia',
            'Oli': 'Oligodendrocyte',
            'OPC': 'OPC',
            'Pericytes': 'Pericyte'})
                                .alias('lenient_cell_type'))
              )
        sc = sc.rename_obs({'Celltype': 'strict_cell_type'})
        if 'obs' in kwargs:
            sc.obs = kwargs['obs']

        # Add IID to obs.
        sc.obs = correct_ids(sc.obs.lazy()).collect()
        sc.save(save_location, overwrite=True, preserve_strings=False)

    return sc


def compute_pseudobulk(single_cell_data: SingleCell, metadata: pl.DataFrame,
                       id_col: str, strict_classification: bool,
                       case_control: bool, save_to: str,
                       retrieve: bool) -> Pseudobulk:
    """
    Computes a pseudobulk expression from the single cell data provided.
    Quality controls based on non-null case control status.
    :param metadata: metadata to consider when performing DE later.
    :param id_col: ID column for each sample.
    :param strict_classification: uses broad cell type classification if
    False else strict cell type classification.
    :param case_control: computes DE and outputs graphs based on case-control
    status later.
    :param retrieve: retrieves existing, QCed Pseudobulk object from save_to if
    True, else loads raw data, QCs, and saves to save_to.
    """
    if retrieve:
        return Pseudobulk(save_to)

    cell_type_column = 'lenient_cell_type' if not strict_classification else 'strict_cell_type'
    pseudobulk = (single_cell_data.pseudobulk(ID_column=id_col,
                                              cell_type_column=cell_type_column,
                                              num_threads=None,
                                              additional_obs=metadata,
                                              QC_column='after_qc'
                                              )
    .qc(
        case_control_column=None if not case_control else 'schizophrenia_status',
        custom_filter=pl.col('schizophrenia_status').is_not_null())
    )
    pseudobulk.save(save_to, overwrite=True)
    return pseudobulk


def differential_expression(pseudobulk: Pseudobulk, covariates: list[str],
                            independent_variable: str, case_control: bool,
                            save_to: str, retrieve: bool) -> DE:
    """
    Computes genes differentially expressed in Pseudobulk provided along with
    statistically important metrics such as FDR, p-values, mean-variance trends,
    etc. Uses covariates provided in regression analysis.
    :param covariates: control variables that may affect differential gene
    expression.
    :param independent_variable: variable that varies, i.e., variable that
    affects differential gene expression and is in the process of being
    investigated.
    :param case_control: outputs DE results (such as graphs for MV trends)
    separated by case-control status if true, else false.
    :param retrieve: retrieves existing DE object from save_to if true,
    else computes DE from scratch and saves to save_to.
    """
    if retrieve:
        return DE(save_to)

    differential_expression = pseudobulk.DE(label_column=independent_variable,
                                            covariate_columns=covariates,
                                            case_control=True if case_control else False)

    differential_expression.plot_voom(save_to, overwrite=True)

    differential_expression.save(save_to, overwrite=True)
    return differential_expression


def get_metadata(metadata_file: str,
                 save_to: str,
                 retrieve: bool,
                 ) -> (
        pl.LazyFrame):
    """
    Loads metadata for single cell expression data. Adds a column named IID that
    combines CMC ID and ID, using CMC ID when it is available and ID
    otherwise. Drops CMC_ID and ID columns. Also converts schizophrenia
    status into a binary encoding (0 for control).

    :param metadata_file: raw metadata file
    :param save_to: where to save combined metadata file to
    :param retrieve: retrieves existing combined metadata from save_to if
    True, else loads metadata, processes and saves to save_to
    """
    if retrieve:
        return pl.scan_csv(save_to, separator='\t')

    metadata = read_columns(metadata_file, delimiter='\t')

    # IID is ID when CMC ID is null and CMC ID otherwise.
    metadata = correct_ids(metadata).lazy().with_columns(
        IID=pl.col('IID').cast(pl.String))

    # Drops CMC_ID, ID columns, creates schizophrenia status column and drops
    # phenotype column.
    metadata = (metadata.drop_nulls(subset='Phenotype').with_columns(
        schizophrenia_status=pl.when(pl.col('Phenotype').eq('CON')).then(0)
        .when(pl.col('Phenotype').eq('SZ')).then(1).otherwise(None))
                .drop('CMC_ID', 'ID', 'Phenotype'))

    (metadata
     .collect()
     .write_csv(save_to, separator='\t'))

    return metadata


def correct_ids(metadata: pl.LazyFrame) -> pl.LazyFrame:
    """
    Corrects ID column by using CMC ID when available else ID.
    """
    metadata = metadata.with_columns(
        IID=pl.when(pl.col('CMC_ID').cast(pl.String).eq(
            pl.lit('NA').cast(pl.String))).then(
            pl.col('ID').cast(pl.Categorical)).otherwise(
            pl.col('CMC_ID').cast(pl.Categorical)))

    return metadata


def correct_cmc_ids(metadata: pl.LazyFrame) -> pl.LazyFrame:
    """
    Correct CMC_ID column by inserting 'CMC_' to the start of every ID. This
    matches the format in genotype metadata.
    """
    metadata = metadata.with_columns(
        CMC_ID=pl.when(
            pl.col('CMC_ID').cast(pl.String) != pl.lit('NA').cast(pl.String))
        .then(pl.lit('CMC_').cast(pl.String) + pl.col('CMC_ID').cast(pl.String))
        .otherwise(pl.lit('NA').cast(pl.String)))
    return metadata


def add_prs_scores(metadata: pl.LazyFrame,
                   prs_file: str) -> pl.LazyFrame:
    """
    Adds PRS scores to metadata file based on IID column in metadata file
    and ID1 column in PRS file.
    """
    metadata = metadata.join(read_columns(prs_file, delimiter='\t',
                                          columns=['ID1', 'Profile_1'], ),
                             how='left', left_on='IID',
                             right_on='ID1').drop_nulls(subset='Profile_1')
    return metadata


def normalise_prs_scores(metadata: pl.DataFrame) -> pl.DataFrame:
    """
    Normalises PRS scores.
    """
    scores = metadata.select('Profile_1')
    mean = scores.mean().item()
    std = scores.std().item()
    prs_scores = metadata.with_columns(
        Profile_1=pl.col('Profile_1').sub(mean).truediv(std))

    return prs_scores


def concatenate_principal_components(pc_files: list[str], save_to: str) -> None:
    """
    Stacks the PCs in each PC file vertically. Intended to combine PCs for
    different cohorts into one file.
    """
    pcs = []
    for pc_file in pc_files:
        pcs.append(read_columns(pc_file, delimiter=','))

    pl.concat(pcs, how='vertical_relaxed').collect().write_csv(save_to,
                                                               separator='\t')


def unique_mapping(df: pl.DataFrame, column: str) -> dict:
    """
    Assigns a number to each unique element of a column in dataframe.
    """
    unique = df.get_column(column).unique()
    mapping = {}
    current_num = 0

    for entry in unique:
        mapping[entry] = current_num
        current_num += 1

    return mapping


def concatenate_prs(prs_files: list[str], save_to: str):
    """
    Stacks the PRS scores in each PC file vertically. Intended to combine PRS
    scores for different cohorts into one file.
    """
    prs = []
    for prs_file in prs_files:
        prs.append(read_columns(prs_file, delimiter='\t'))

    all_prs = pl.concat(prs, how='vertical_relaxed').collect()
    all_prs.write_csv(save_to,
                      separator='\t')


if __name__ == "__main__":
    ### LOAD SINGLE CELL DATA ###
    single_cell = load_single_cell_data(
        f'{SCHIZO_DIR_SINGLE_CELL_RAW}/combinedCells_ACTIONet.h5ad', True,
        f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_data.h5ad')

    # Explicit cast required because of incorrect inference.
    single_cell.obs = single_cell.obs.with_columns(
        IID=pl.col('IID').cast(pl.String))

    ### LOAD AND PROCESS METADATA ###
    # A couple of basic null filters.
    if not os.path.exists(
            f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata_compounded'):
        if not os.path.exists(f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata'):
            single_cell_metadata = get_metadata(
                f'{SCHIZO_DIR_SINGLE_CELL_RAW}/individual_metadata_filtered_grouped.tsv',
                f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata', False)
        else:
            single_cell_metadata = get_metadata(
                f'{SCHIZO_DIR_SINGLE_CELL_RAW}/individual_metadata_filtered_grouped.tsv',
                f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata', True)

        # Compute, process, and add PCs as covariates for DE.
        if not os.path.exists(
                f'{SCHIZO_DIR_SINGLE_CELL}/mssm_principle_components.eigenvec'):
            get_principle_components(
                f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mssm_genotypes_qc', 10,
                f'{SCHIZO_DIR_SINGLE_CELL}/mssm_principle_components')

        if not os.path.exists(
                f'{SCHIZO_DIR_SINGLE_CELL}/mclean_principle_components.eigenvec'):
            get_principle_components(
                f'{SCHIZO_DIR_GWAS}/cleaned_genotypes/mclean_genotypes_qc', 10,
                f'{SCHIZO_DIR_SINGLE_CELL}/mclean_principle_components')

        if not os.path.exists(f'{SCHIZO_DIR_SINGLE_CELL}/principal_components'):
            concatenate_principal_components(
                [
                    f'{SCHIZO_DIR_SINGLE_CELL}/mclean_principle_components.eigenvec',
                    f'{SCHIZO_DIR_SINGLE_CELL}/mssm_principle_components.eigenvec'],
                f'{SCHIZO_DIR_SINGLE_CELL}/principal_components')

        single_cell_metadata = add_principal_components_to_metadata(
            single_cell_metadata,
            f'{SCHIZO_DIR_SINGLE_CELL}/principal_components',
            10, 'IID', 'IID')

        # Add PRS scores.
        if not os.path.exists(f'{SCHIZO_DIR_SINGLE_CELL}/prs'):
            concatenate_prs(
                [
                    f'{SCHIZO_DIR_PRS}/mssm_prs.profile',
                    f'{SCHIZO_DIR_PRS}/mclean_prs.profile'],
                f'{SCHIZO_DIR_SINGLE_CELL}/prs')

        single_cell_metadata = add_prs_scores(single_cell_metadata,
                                              f'{SCHIZO_DIR_SINGLE_CELL}/prs')

        single_cell_metadata = single_cell_metadata.drop_nulls(
            subset=['schizophrenia_status', 'Age', 'Gender', 'Batch'])

        single_cell_metadata = single_cell_metadata.with_columns(
            PMI=pl.col('PMI').fill_null(strategy='mean'))

        single_cell_metadata = single_cell_metadata.collect()

        # Convert covariates to enum data type by uncommenting code below.
        # single_cell_metadata_final = single_cell_metadata_final.with_columns(
        #     Gender=pl.col('Gender').cast(pl.Enum(
        #         single_cell_metadata_final.get_column('Gender').unique())),
        #     Batch=pl.col('Batch').cast(pl.Enum(
        #         single_cell_metadata_final.get_column('Batch').unique())))

        # Convert to integer dtype by uncommenting code below.
        gender_mapping = unique_mapping(single_cell_metadata, 'Gender')
        single_cell_metadata = single_cell_metadata.with_columns(
            Gender=
            pl.col('Gender').replace(gender_mapping).cast(pl.Int64))
        batch_mapping = unique_mapping(single_cell_metadata, 'Batch')
        single_cell_metadata = single_cell_metadata.with_columns(
            Batch=pl.col('Batch').replace(batch_mapping).cast(pl.Int64))

        single_cell_metadata = normalise_prs_scores(single_cell_metadata)

        single_cell_metadata_final = single_cell_metadata

        single_cell_metadata_final.write_csv(
            f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata_compounded',
            separator='\t')

        # Reset for convenience's sake.
        single_cell_metadata = single_cell_metadata_final
    else:
        single_cell_metadata = pl.read_csv(
            f'{SCHIZO_DIR_SINGLE_CELL}/single_cell_metadata_compounded',
            separator='\t')

    ### PSEUDOBULK ###
    pseuobulk_strict_prs = compute_pseudobulk(single_cell,
                                              single_cell_metadata,
                                              'IID', True,
                                              False,
                                              f'{SCHIZO_DIR_SINGLE_CELL}/pb_prs_strict',
                                              True)

    pseuobulk_lenient_prs = compute_pseudobulk(single_cell,
                                               single_cell_metadata,
                                               'IID',
                                               False,
                                               False,
                                               f'{SCHIZO_DIR_SINGLE_CELL}/pb_prs_lenient',
                                               True)

    pseuobulk_strict_cc = compute_pseudobulk(single_cell,
                                             single_cell_metadata,
                                             'IID', True,
                                             True,
                                             f'{SCHIZO_DIR_SINGLE_CELL}/pb_case_control_strict',
                                             True)

    pseuobulk_lenient_cc = compute_pseudobulk(single_cell,
                                              single_cell_metadata, 'IID',
                                              False,
                                              True,
                                              f'{SCHIZO_DIR_SINGLE_CELL}/pb_case_control_lenient',
                                              True)

    ### DIFFERENTIAL EXPRESSION ###
    # For CC.
    covariates_case_control = ['Batch', 'Gender', 'Age', 'PMI'] + [f'PC{num}'
                                                                   for num in
                                                                   range(1, 11)]

    differential_expression_strict_cc = differential_expression(
        pseuobulk_strict_cc,
        covariates_case_control,
        'schizophrenia_status',
        True,
        f'{SCHIZO_DIR_SINGLE_CELL}/de_case_control_strict',
        True)

    differential_expression_lenient_cc = differential_expression(
        pseuobulk_lenient_cc,
        covariates_case_control,
        'schizophrenia_status',
        True,
        f'{SCHIZO_DIR_SINGLE_CELL}/de_case_control_lenient',
        True)

    # Get raw hits.
    strict_cc_hits: pl.DataFrame = (
        differential_expression_strict_cc.get_hits())
    lenient_cc_hits: pl.DataFrame = (
        differential_expression_lenient_cc.get_hits())

    covariates_prs = ['Batch', 'Gender', 'Age', 'PMI',
                      'schizophrenia_status'] + [f'PC{num}'
                                                 for num in
                                                 range(1,
                                                       11)]
    # Repeat for PRS.
    differential_expression_strict_prs = differential_expression(
        pseuobulk_strict_prs,
        covariates_prs,
        'Profile_1',
        False,
        f'{SCHIZO_DIR_SINGLE_CELL}/de_prs_strict',
        True)

    differential_expression_lenient_prs = differential_expression(
        pseuobulk_lenient_prs,
        covariates_prs,
        'Profile_1',
        False,
        f'{SCHIZO_DIR_SINGLE_CELL}/de_prs_lenient',
        True)

    strict_prs_hits: pl.DataFrame = (
        differential_expression_strict_prs.get_hits())
    lenient_prs_hits: pl.DataFrame = (
        differential_expression_lenient_prs.get_hits())

    # Compute overlap between CC and PRS hits.
    lenient_overlaps = compute_overlaps([lenient_prs_hits,
                                         lenient_cc_hits])
    lenient_overlapping_genes = lenient_overlaps.get_column('gene')
    lenient_only_cc = lenient_cc_hits.filter(
        pl.col('gene').is_in(lenient_overlapping_genes).not_())
    lenient_only_prs = lenient_prs_hits.filter(
        pl.col('gene').is_in(lenient_overlapping_genes).not_())

    logger.info(
        f'[Lenient] Overlaps between PRS and CC:\n'
        f'{lenient_overlaps.get_column('gene').to_list()}')
    print('\n')
    logger.info(
        f'[Lenient] Only CC:\n'
        f'{lenient_only_cc.get_column('gene').to_list()}')
    print('\n')
    logger.info(
        f'[Lenient] Only PRS:\n'
        f'{lenient_only_prs.get_column('gene').to_list()}')
    print('\n\n')

    strict_overlaps = compute_overlaps([strict_cc_hits,
                                        strict_prs_hits])
    strict_overlapping_genes = strict_overlaps.get_column('gene')
    strict_only_cc = strict_cc_hits.filter(
        pl.col('gene').is_in(strict_overlapping_genes).not_())
    strict_only_prs = strict_prs_hits.filter(
        pl.col('gene').is_in(strict_overlapping_genes).not_())

    logger.info(
        f'[Strict] Overlaps between PRS and CC:\n'
        f'{strict_overlaps.get_column('gene').to_list()}')
    print('\n')
    logger.info(
        f'[Strict] Only CC:\n'
        f'{strict_only_cc.get_column('gene').to_list()}')
    print('\n')
    logger.info(
        f'[Strict] Only PRS:\n'
        f'{strict_only_prs.get_column('gene').to_list()}')

    # Compute a more comprehensive overlap between CC and PRS hits. Lists:
    # 1. Number of overlapping hits, non-overlapping hits per cell type
    # 2. Hits grouped by cell type and overlap status (in that order)
    strict_all_hits = pl.concat([strict_only_cc.with_columns(
        overlap_status=pl.lit('Only CC')).select('cell_type', 'gene',
                                                 'overlap_status', 'FDR'),
                                 strict_overlaps.with_columns(
                                     overlap_status=pl.lit(
                                         'CC and PRS')).select('cell_type',
                                                               'gene',
                                                               'overlap_status',
                                                               'FDR'),
                                 strict_only_prs.with_columns(
                                     overlap_status=pl.lit('Only PRS')).select(
                                     'cell_type', 'gene',
                                     'overlap_status', 'FDR')])

    lenient_all_hits = pl.concat([lenient_only_cc.with_columns(
        overlap_status=pl.lit('Only CC')).select('cell_type', 'gene',
                                                 'overlap_status', 'FDR'),
                                  lenient_overlaps.with_columns(
                                      overlap_status=pl.lit(
                                          'CC and PRS')).select('cell_type',
                                                                'gene',
                                                                'overlap_status',
                                                                'FDR'),
                                  lenient_only_prs.with_columns(
                                      overlap_status=pl.lit('Only PRS')).select(
                                      'cell_type', 'gene',
                                      'overlap_status', 'FDR')])

    # Make all rows visible for debugging.
    pl.Config.set_tbl_rows(1000)

    # Cell type, then overlap statuses hierarchy. Could go reverse, but there
    # was a variable naming issue where I called group_by on the wrong df so
    # I tried this method and honestly it works better anyway.
    print('\n\n\n')

    logger.info(f'[Lenient] Detailed:')
    for cell_type, data in lenient_all_hits.group_by('cell_type'):
        logger.info(f'{cell_type[0]}:')
        for overlap, data2 in data.group_by('overlap_status'):
            logger.info(f'{overlap[0]}:')
            logger.info(f'{data2.select('gene', 'FDR')}')
            print('\n')
        print('\n')

    print('\n\n\n')
    logger.info(f'[Strict] Detailed:')
    for cell_type, data in strict_all_hits.group_by('cell_type'):
        logger.info(f'{cell_type[0]}:')
        for overlap, data2 in data.group_by('overlap_status'):
            logger.info(f'{overlap[0]}:')
            logger.info(f'{data2.select('gene', 'FDR')}')
            print('\n')
        print('\n')

    num_hits_list_strict = []
    num_hits_list_lenient = []

    for cell_type in lenient_all_hits.get_column('cell_type').unique():
        cell_type_entries = lenient_all_hits.filter(pl.col(
            'cell_type').eq(cell_type))
        num_hits_dict = {'Cell Type': cell_type}

        num_hits_dict['Only PRS'] = cell_type_entries.filter(
            pl.col('overlap_status').eq('Only PRS')).select(pl.len()).item()
        num_hits_dict['Only CC'] = cell_type_entries.filter(
            pl.col('overlap_status').eq('Only CC')).select(pl.len()).item()
        num_hits_dict['CC and PRS'] = cell_type_entries.filter(
            pl.col('overlap_status').eq('CC and PRS')).select(pl.len()).item()

        num_hits_list_lenient.append(num_hits_dict)

    for cell_type in strict_all_hits.get_column('cell_type').unique():
        cell_type_entries = strict_all_hits.filter(pl.col(
            'cell_type').eq(cell_type))
        num_hits_dict = {'Cell Type': cell_type}
        num_hits_dict['Only PRS'] = cell_type_entries.filter(pl.col(
            'overlap_status').eq('Only PRS')).select(pl.len()).item()
        num_hits_dict['Only CC'] = cell_type_entries.filter(pl.col(
            'overlap_status').eq('Only CC')).select(pl.len()).item()
        num_hits_dict['CC and PRS'] = cell_type_entries.filter(pl.col(
            'overlap_status').eq('CC and PRS')).select(pl.len()).item()

        num_hits_list_strict.append(num_hits_dict)

    num_hits_strict = pl.DataFrame(data=num_hits_list_strict,
                                   schema={'Cell Type': pl.String,
                                           'Only CC': pl.Int64,
                                           'CC and PRS': pl.Int64,
                                           'Only PRS': pl.Int64})
    num_hits_lenient = pl.DataFrame(data=num_hits_list_lenient,
                                    schema={'Cell Type': pl.String,
                                            'Only CC': pl.Int64,
                                            'CC and PRS': pl.Int64,
                                            'Only PRS': pl.Int64})

    logger.info(f'[Strict] Number of Hits:\n'
                f'{num_hits_strict}')
    logger.info(f'[Lenient] Number of Hits:\n'
                f'{num_hits_lenient}')

    # TODO: Filter to white people in MSSM.

    # Compute AUC scores restricted to people for whom PRS scores were computed.
    if not os.path.exists(f'{SCHIZO_DIR_SINGLE_CELL}/prs_dx_swarm.pdf'):
        graph_prs_disease_status(single_cell_metadata, 'Profile_1',
                                 'schizophrenia_status', 'swarm',
                                 f'{SCHIZO_DIR_SINGLE_CELL}/prs_dx_swarm.pdf')
    if not os.path.exists(f'{SCHIZO_DIR_SINGLE_CELL}/prs_dx_box.pdf'):
        graph_prs_disease_status(single_cell_metadata, 'Profile_1',
                                 'schizophrenia_status', 'box',
                                 f'{SCHIZO_DIR_SINGLE_CELL}/prs_dx_box.pdf')

    logger.info(
        f'AUC score for PRSs - '
        f'{roc_auc_score(
            single_cell_metadata.get_column('schizophrenia_status'),
            single_cell_metadata.get_column('Profile_1'))}')

    # Uncomment to save plots of PRS against CC status.
    # if not os.path.exists(f'{SCHIZO_DIR_SINGLE_CELL}/prs_dx_swarm.pdf'):
    #     graph_prs_disease_status(single_cell_metadata, 'Profile_1',
    #                              'schizophrenia_status', 'swarm',
    #                              f'{SCHIZO_DIR_SINGLE_CELL}/prs_dx_swarm.pdf')
    # if not os.path.exists(f'{SCHIZO_DIR_SINGLE_CELL}/prs_dx_box.pdf'):
    #     graph_prs_disease_status(single_cell_metadata, 'Profile_1',
    #                              'schizophrenia_status', 'box',
    #                              f'{SCHIZO_DIR_SINGLE_CELL}/prs_dx_box.pdf')
