from utils import run
from constants import *


PROJECT_DIR = 'projects/def-wainberg'

run(f'{PLINK2} '
    f'--bfile {PROJECT_DIR}/jeesonja/cleaned_genome/'
    f'{PLINK_OUTPUT_FILE}_merged_qc_hetero '
    f'--score {PROJECT_DIR}/jeesonja/scores/prs_weights_without_apoe.txt 1 4 6 '
    f'header '
    f'list-variants')