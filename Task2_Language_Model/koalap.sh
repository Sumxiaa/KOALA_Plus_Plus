#!/bin/bash
python run_exp.py --layers 4 --batch_size 32 --lr 0.002 --sigma 0.1 --q 0.1 --weight_decay 0.0001 --log 1 --epochs 50 --optimizer KOALAPlusPlus \
--device cuda


