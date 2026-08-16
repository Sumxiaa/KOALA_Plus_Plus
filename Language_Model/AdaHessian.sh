#!/bin/bash
python run_exp.py --layers 4 --batch_size 32 --weight_decay 0.1 --lr 0.15 --log 1 --epochs 50 --optimizer AdaHessian --device cuda