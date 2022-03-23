#!/bin/bash
conda activate hd
begin=0
end=14
i=0
dir="/home/mandonaire/scripts/database_run"
# 0 al 14, 15 al 29, 29 al 43, del 43 al 57, del 57 al 72, del 72 al 86, del 86 al 91
for FOLDER in /data/mandonaire/senamhi-huaraz/ftp_huaraz/*;
do
    if [ $i -ge $begin ] && [ $i -le $end ]; then 
        echo procesando $FOLDER 
        nohup python preprocessing_py3.py --folder $FOLDER > "${dir}/${i}.out" &
    fi
    ((i++))
done 
