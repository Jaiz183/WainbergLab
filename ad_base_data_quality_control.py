import polars as pl

from utils import get_coding_genes
from constants import *

PROJECT_DIR = 'projects/def-wainberg'

# Get gene data with loci from PRS file.
gene_mapping = get_coding_genes(genome_build='hg19', gencode_version=46,
                                return_file=False)
prs_data = pl.scan_csv(f'{PROJECT_DIR}/jeesonja/prs_weights.txt',
                       separator='\t', truncate_ragged_lines=True, skip_rows=19)
# print(gene_mapping.columns)
# print(prs_data.columns)

apoe_mapping = gene_mapping.filter(pl.col('gene').str.contains('APOE'))
apoe_loc_start = apoe_mapping.select('start')[0, 0]
apoe_loc_end = apoe_mapping.select('end')[0, 0]
chrom = int(apoe_mapping.select('chrom')[0, 0][3:])

# assert chrom is not None and apoe_loc_start is not None and apoe_loc_end is not None

low_bd = apoe_loc_start - REGION
# print(low_bd)
up_bd = apoe_loc_end + REGION
# print(up_bd)

# Remove rows corresponding to genes 1 megabase within APOE gene. Try
# using start and end fields from DF returned.
# fix filtering.
prs_without_apoe = prs_data.filter(
    pl.col('chr_position').le(low_bd) | pl.col('chr_position').ge(up_bd))

# Separately compute non 19 chromosome data and 19 chromosome data outside APOE region, then concatenate.
print(
    f'Number of entries removed: {prs_data.select(pl.len()).collect().item() - prs_without_apoe.select(pl.len()).collect().item()}')

final_prs = prs_without_apoe.collect()
final_prs.write_csv(f'{PROJECT_DIR}/jeesonja/prs_weights_without_apoe.txt',
                    separator='\t', include_header=True)
print(final_prs)
