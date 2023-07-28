# run this script from the root (revolve folder): ./experiments/default_study/run-analysis.sh PARAMSFILE

DIR="$(dirname "${BASH_SOURCE[0]}")"
study_path="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=$DIR/paramsdefault.sh
  else
    params_file=$1
fi

source $params_file

comparison="all"
python3 experiments/default_study/analyze_robutsness.py $study $experiments $runs $generations $outputs_path $simulator $loop $body_phenotype $bisymmetry $comparison;
