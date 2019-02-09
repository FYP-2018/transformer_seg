#!/bin/bash
#PBS -N T_SEG
# PBS -l walltime=24:00:00
#PBS -e ./error.txt
#PBS -o ./output.txt

cd /home/u20866/fyp/transformer_seg
echo Starting calculation
python main.py
echo End of calculation
