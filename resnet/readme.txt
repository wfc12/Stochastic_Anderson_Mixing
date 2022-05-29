export CUDA_VISIBLE_DEVICES=1 && python -u trainer.py --arch resnet20 --save-dir save_adasam_resnet20 --optim adasam --epoch 160 --batch-size 128 --lr 0.1 --momentum 0 --period 1 --hist_length 10 --damp 0.01 --beta 1 --alpha 1 --weight_decay 0.0015 --seed 1 > log_adasam_resnet20 &&  python -u showprec.py

