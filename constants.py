# System constants.
PROJECT_DIR = 'projects/def-wainberg'
CONTROLLED_ACCESS = f'projects/rrg-wainberg/CommonMind/ControlledAccess'
SCRATCH_DIR = f'scratch'

# Command line tools
BCF = f'bcftools'
NORMALISER = f'norm'
PLINK2 = 'plink2'
PLINK = 'plink'
AWK = f'awk'
GCTA = f'gcta64'
GCTB = f'gctb'
LDAK = './ldak5.2.linux '
NGT = 'NGT/build/bin/ngt/ngt'

### kNN ###
KNN_DIR = f'{PROJECT_DIR}/jeesonja/knn'
LABEL_TRANSFER = 'lt'
UMAP = 'u'
VALIDATION_DATASET = 'scratch/validation.npy'
TRAINING_DATASET = 'scratch/training.npy'
TRUE_NEIGHBOURS = 'scratch/true_nn.npy'


### Schizoprhenia ###
SCHIZO_DIR_SINGLE_CELL = f'{PROJECT_DIR}/jeesonja/schizo/single_cell'
SCHIZO_DIR_GWAS = f'{PROJECT_DIR}/jeesonja/schizo/gwas'
SCHIZO_DIR_SINGLE_CELL_RAW = f'{PROJECT_DIR}/single-cell/SZBDMulticohort'
SCHIZO_DIR_CMC_GENOTYPES = f'{CONTROLLED_ACCESS}/Data/Genotypes/SNPs/Release3/Imputed/MSSM-Penn-Pitt'
SCHIZO_CMC_METADATA_DIR = f'{CONTROLLED_ACCESS}/Data/Genotypes/SNPs/Release3/Metadata'
SCHIZO_DE_DIR = f'{PROJECT_DIR}/jeesonja/schizo/single_cell/differential_expression'
SCHIZO_DIR_PRS = f'{PROJECT_DIR}/jeesonja/schizo/prs'

SCHIZO_SINGLE_CELL_FILE = 'combinedCells_ACTIONet.h5ad'
SCHIZO_GENOTYPE_FILE = 'SZBDMulti-Seq_Cohort_Genotypes.vcf.gz'
SCHIZO_METADATA_FILE = 'individual_metadata.tsv'

### AD ###
# Single cell analysis, AD.
# File names.
PSEUDO_BULK_DIR = 'pseudobulk'
SINGLE_CELL_DIR = f'{PROJECT_DIR}/jeesonja/single_cell'
DE_DIR = f'{SINGLE_CELL_DIR}/differential_expression'

DE_FILE = 'de_data'
DE_PLOT = 'de_plot'
SCORES_FILE = 'plink2.sscore'
SCORES_FILE_NORMALISED = 'prs_normalised'
RACE_FILTERED_DATA = f'{PROJECT_DIR}/jeesonja/cleaned_genome/race_filtered_data'
RACE_FILTERED_DATA_LD_PRUNED = f'{PROJECT_DIR}/jeesonja/cleaned_genome/race_filtered_data_pruned'
RACE_FILTERED_DATA_NO_MISSING_IDS = f'{PROJECT_DIR}/jeesonja/cleaned_genome/race_filtered_data_no_missing_ids'
CAUCASIAN_SAMPLES = f'{PROJECT_DIR}/jeesonja/cleaned_genome/caucasian_samples'
ROSMAP_METADATA_FILE = f'{SINGLE_CELL_DIR}/Green/dataset_978_basic_04-21-2023_with_pmAD.csv'
ROSMAP_ID_FILE = f'{SINGLE_CELL_DIR}/rosmap_ids'
METADATA_IDS_FILE = f'{SINGLE_CELL_DIR}/rosmap_metadata_ids'
SINGLE_CELL_QC = 'single_cell.h5ad'
PC_FILE = f'{SINGLE_CELL_DIR}/pc_caucasian'

# Quality control for PRS scoring.
PLINK_OUTPUT_FILE = 'cleaned_target_data'
VCF_OUTPUT_FILE = 'multiallelic_split_target_data'
HETEROZYGOSITY_OUTPUT_FILE = f'heterozygosity_data'
VALID_HETEROZYGOSITY_OUTPUT_FILE = f'valid_heterozygosity_data'
QC_TARGET_DATA_WITHOUT_SCORES = f'{PROJECT_DIR}/jeesonja/cleaned_genome/{PLINK_OUTPUT_FILE}_merged_qc_hetero'

# APOE removal.
START_LOC = 1
END_LOC = 2
GENE = 4
CHROM = 0
REGION = 1_000_000

# Columns.
PROJ_ID = 'projid'
WGS_ID = 'wgs_id'
FAMILY_ID = '#FID'
INDIVIDUAL_ID = 'IID'
SCORE_COL = 'SCORE1_AVG'
RACE_COL = 'race7'
RACES = {'caucasian': 1}
