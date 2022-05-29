export CUDA_VISIBLE_DEVICES=0 && python -u main.py --cuda --optim padasam --epoch 200 --dropouti 0.4 --dropouth 0.25 --batch_size 20 --data <dataset> --lr 0.005 --momentum 0 --period 1 --hist_length 10 --damp 0.01 --beta 1 --alpha 1 --seed 1 > log_padasam &&  python -u showprec.py

