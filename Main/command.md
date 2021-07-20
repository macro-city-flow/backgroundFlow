## configuration

```bash
conda activate torch
```

## basic running

```bash
python3 main.py --data chengdu --model-name GCN --settings forecast --data-module NS --lr 0.00114514 --weight-decay 1.5e-3 --loss mse --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --hidden-dim 64 --output-dim 25 --gpus 1
```

```bash
python3 main.py --data chengdu --model-name GRU --settings forecast --data-module S --lr 0.00114514 --weight-decay 1.5e-3 --loss mse --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --hidden-dim 64 --output-dim 25 --gamma 0.6 --gradient-clip-val 5 --gpus 1
```

```bash
python3 main.py --data chengdu --model-name TGCN --settings forecast --data-module S --lr 0.00114514 --weight-decay 1.5e-3 --loss mse --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --hidden-dim 64 --output-dim 25 --gamma 0.5 --gradient-clip-val 5 --gpus 1
```

```bash
python3 main.py --data chengdu --model-name MDN --settings densityForecast --data-module NS --lr 0.00114514 --weight-decay 1.5e-3 --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --gamma 5 --output-dim 25 --gradient-clip-val 30 --gpus 1
```



### Test needed

```bash
python3 main.py --data chengdu --model-name ChebyNet --settings forecast --data-module NS --lr 0.00114514 --weight-decay 1.5e-3 --loss mse --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --hidden-dim 64 --output-dim 25 --gpus 1
```


## monitor metrics

```bash
tensorboard --logdir [LOG DIR PATH]
```

## batch running

<!-- Not implied yet -->