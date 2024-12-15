#!/bin/zsh
rsync -zav --progress constants.py useful_functions.py ad_base_data_quality_control.py ad_target_data_quality_control.py ad_prs_scoring.py ad_single_cell_analysis.py schizo_single_cell_analysis.py schizo_single_cell_analysis.r schizo_gwas_analysis.py schizo_prs_scoring.py n:~/
rsync -zav --progress knn_profile.py faiss_profile.py n:~/
