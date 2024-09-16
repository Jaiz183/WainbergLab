from utils import load_dbSNP, get_rs_numbers_bim_or_pvar
from constants import *

print(f"Converting to plink 1 format...")

# convert to separate chrom. files without QC.
chromosomes = list(range(1, 23))
# for chromosome in chromosomes:
#     # split multi-allelic variants. Include indels.
#     # run(f'{BCF} {NORMALISER} '
#     #     f'--multiallelics -any '
#     #     f'--do-not-normalize '
#     #     f'--output {PROJECT_DIR}/jeesonja/split_multiallelics/{VCF_OUTPUT_FILE}_{chromosome}.vcf.gz '
#     #     f'--output-type z '
#     #     f'{PROJECT_DIR}/wgs/AMP-AD/NIA_JG_1898_samples_GRM_WGS_b37_'
#     #     f'JointAnalysis01_2017-12-08_{chromosome}.recalibrated_variants.vcf.gz')
#
#     run(f'{PLINK} '
#         f'--vcf {PROJECT_DIR}/wgs/AMP-AD/NIA_JG_1898_samples_GRM_WGS_b37_'
#         f'JointAnalysis01_2017-12-08_{chromosome}.recalibrated_variants.vcf.gz '
#         f'--vcf-min-gq 20 '
#         f'--vcf-min-dp 10 '
#         f'--vcf-min-qual 30 '
#         f'--max-alleles 2 '
#         f'--double-id '
#         f'--make-bed '
#         f'--out {PROJECT_DIR}/jeesonja/cleaned_chromosomes/{PLINK_OUTPUT_FILE}_{chromosome}')

print("Merging chromosomal files...")
# merge files into full genome PLINK file.
# bfiles = [f'{PROJECT_DIR}/jeesonja/cleaned_chromosomes/{PLINK_OUTPUT_FILE}_{chromosome}' for chromosome in chromosomes]
# merge_bfiles(bfiles, f'{PROJECT_DIR}/jeesonja/cleaned_genome/{PLINK_OUTPUT_FILE}_merged')

print("Quality controlling merged data...")
# conduct QC on final PLINK file.
# run(f'{PLINK} '
#     f'--bfile {PROJECT_DIR}/jeesonja/cleaned_genome/{PLINK_OUTPUT_FILE}_merged '
#     f'--make-bed '
#     f'--double-id '
#     f'--max-maf 0.99 '
#     f'--maf 0.01 '
#     f'--geno '
#     f'--hwe 1e-15 midp '
#     f'--mind '
#     f'--out {PROJECT_DIR}/jeesonja/cleaned_genome/{PLINK_OUTPUT_FILE}_merged_qc')

# Filter for heterozygosity.
# Calculate heterozygosity rates.
print(f'Calculating heterozygosity rates...')
# run(f'{PLINK} '
#     f'--bfile {PROJECT_DIR}/jeesonja/cleaned_genome/{PLINK_OUTPUT_FILE}_merged_qc '
#     f'--het '
#     f'--keep {PROJECT_DIR}/jeesonja/cleaned_genome/{PLINK_OUTPUT_FILE}_merged_qc.fam '
#     f'--out {PROJECT_DIR}/jeesonja/cleaned_genome/{HETEROZYGOSITY_OUTPUT_FILE}')

# Remove individuals with F coefficients that are more than 3 standard.
# deviation (SD) units from the mean.
print(f"Computing individuals with high heterozygosity...")

# I believe that a het file is comma separated.
# data = pl.scan_csv(f'{HETEROZYGOSITY_OUTPUT_FILE}.het', separator='\t',
#                    dtypes={'#FID': str, 'IID': str})
# f_coeff = data.select("F")
# mean = f_coeff.mean().collect()[0, 0]
# std = f_coeff.std().collect()[0, 0]
# print(mean, std)
# upper_bd = mean + 3 * std
# lower_bd = mean - 3 * std
# # replace with filter.
# valid = data.filter(pl.col("F") <= upper_bd, pl.col("F") >= lower_bd)
# valid.collect().write_csv(
#     f'{PROJECT_DIR}/jeesonja/cleaned_genome/{VALID_HETEROZYGOSITY_OUTPUT_FILE}',
#     separator="\t",
#     include_header=True)
# print(
#     f'Number of entries removed: {data.select(pl.len()).collect().item() - valid.select(pl.len()).collect().item()}')

# Remove all anomalous heterozygosity entries from main file.
# run(f'{PLINK} '
#     f'--bfile {PROJECT_DIR}/jeesonja/cleaned_genome/{PLINK_OUTPUT_FILE}_merged_qc '
#     f'--keep {PROJECT_DIR}/jeesonja/cleaned_genome/{VALID_HETEROZYGOSITY_OUTPUT_FILE} '
#     f'--make-bed '
#     f'--out {PROJECT_DIR}/jeesonja/cleaned_genome/{PLINK_OUTPUT_FILE}_merged_qc_hetero ')

# Add variant IDs to BIM file because they were missing.
genome_rs_numbers = load_dbSNP('hg19')
get_rs_numbers_bim_or_pvar(
    f'{PROJECT_DIR}/jeesonja/cleaned_genome/cleaned_target_data_merged_qc_hetero.bim',
    genome_rs_numbers, verbose=True)
