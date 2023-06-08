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


#TODO: ugly. make setup sh account for commas instead
experimentscoma=""
bisymmetrycoma=""
for experiment in "${experiments[@]}"
do
  experimentscoma=("${experimentscoma}${experiment},")
done
experimentscoma=${experimentscoma::-1}
for bisy in "${bisymmetry[@]}"
do
  bisymmetrycoma=("${bisymmetrycoma}${bisy},")
done
bisymmetrycoma=${bisymmetrycoma::-1}



file="${outputs_path}/${study}/analysis/video_bests.mpg";

printf " \n making video..."
screen -d -m -S ${study}_videos ffmpeg -f x11grab -r 25 -i :1 -qscale 0 $file;
python3 experiments/${study_path}/watch_robots.py $study $experimentscoma $watchruns $generations $outputs_path $simulator $loop $body_phenotype $bisymmetrycoma;

pkill -f ${study}_videos
pkill -f ${study}_loop

printf " \n finished video!"