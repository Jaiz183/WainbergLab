#!/bin/bash
if [[ "$1" == "n" ]]; then
  echo "Syncing to only Narval..."
  rsync -zav --progress constants.py useful_functions.py ad_base_data_quality_control.py ad_target_data_quality_control.py ad_prs_scoring.py ad_single_cell_analysis.py schizo_single_cell_analysis.py schizo_gwas_analysis.py schizo_prs_scoring.py n:~/
  rsync -zav --progress knn_profile.py faiss_profile.py faiss_profiling_mre.py n:~/
  rsync -zav --progress clustering.py n:~/
elif [[ "$1" == "s" ]]; then
  echo "Syncing to only Niagara..."
  rsync -zav --progress constants.py useful_functions.py ad_base_data_quality_control.py ad_target_data_quality_control.py ad_prs_scoring.py ad_single_cell_analysis.py schizo_single_cell_analysis.py schizo_gwas_analysis.py schizo_prs_scoring.py s:~/
  rsync -zav --progress knn_profile.py faiss_profile.py faiss_profiling_mre.py s:~/
  rsync -zav --progress clustering.py s:~/
else
  echo "Syncing to Narval and Niagara..."
  rsync -zav --progress constants.py useful_functions.py ad_base_data_quality_control.py ad_target_data_quality_control.py ad_prs_scoring.py ad_single_cell_analysis.py schizo_single_cell_analysis.py schizo_gwas_analysis.py schizo_prs_scoring.py n:~/
  rsync -zav --progress knn_profile.py faiss_profile.py faiss_profiling_mre.py n:~/
  rsync -zav --progress clustering.py n:~/

  rsync -zav --progress constants.py useful_functions.py ad_base_data_quality_control.py ad_target_data_quality_control.py ad_prs_scoring.py ad_single_cell_analysis.py schizo_single_cell_analysis.py schizo_gwas_analysis.py schizo_prs_scoring.py s:~/
  rsync -zav --progress knn_profile.py faiss_profile.py faiss_profiling_mre.py s:~/
  rsync -zav --progress clustering.py s:~/
fi
