#!/bin/zsh
if [[ "$1" == "n" ]]; then
	rsync -zav n:~/jeesonja/knn/faiss/ ./faiss_results/
	rsync -zav n:~/jeesonja/clustering/results/ ./clustering_results/
elif [[ "$1" == "s" ]]; then
	SCRATCH=/scratch/w/wainberg
	rsync -zav s:$SCRATCH/jeesonja/knn/faiss/ ./faiss_results/
        rsync -zav s:$SCRATCH/jeesonja/clustering/results/ ./clustering_results/
else;
	echo "Option $1 not recognized. Use s for Niagara and n for Narval."
fi
