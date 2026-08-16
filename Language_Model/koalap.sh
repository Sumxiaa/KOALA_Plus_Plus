#!/bin/bash
python run_exp.py --layers 4 --batch_size 32 --lr 2e-3 --sigma 0.1 --q 0.1 --weight_decay 8e-5 --log 1 --epochs 50 --optimizer KOALAPlusPlus \
--device cuda


