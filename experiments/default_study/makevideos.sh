DIR="$(dirname "${BASH_SOURCE[0]}")"
study_path="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=paramsdefault
  else
    params_file=$1
fi

source $DIR/$params_file.sh

file="${outputs_path}/${study}/analysis/video_bests.mpg";

printf " \n making video..."
screen -d -m -S videos ffmpeg -f x11grab -r 25 -i :1 -qscale 0 $file;
python3 experiments/${study_path}/watch_robots.py $study $experiments $runs $generations $outputs_path $simulator $loop $body_phenotype;
killall screen
printf " \n finished video!"