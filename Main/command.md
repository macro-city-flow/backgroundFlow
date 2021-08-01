# commands

## basic running

```bash
python3 main.py --data-path data/chengdu_data.pkl --adj-path data/chengdu_adj.pkl  --model-name GCN --settings forecast --data-module NS --lr 0.00114514 --weight-decay 1.5e-3 --loss mse --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --hidden-dim 64 --input-dim 25 --output-dim 25 --feature-dim 25 --gpus 1
```

```bash
python3 main.py --data-path data/chengdu_data.pkl --adj-path data/chengdu_adj.pkl --model-name GRU --settings forecast --data-module S --lr 0.00114514 --weight-decay 1.5e-3 --loss mse --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --hidden-dim 64 --input-dim 25 --output-dim 25 --feature-dim 25 --gamma 0.6 --gradient-clip-val 5 --gpus 1
```

```bash
python3 main.py --data-path data/chengdu_data.pkl --adj-path data/chengdu_adj.pkl --model-name TGCN --settings forecast --data-module S --lr 0.00114514 --weight-decay 1.5e-3 --loss mse --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --hidden-dim 64 --input-dim 25 --output-dim 25 --feature-dim 25 --gamma 0.5 --gradient-clip-val 5 --gpus 1
```

```bash
python3 main.py --data-path data/chengdu_data.pkl --adj-path data/chengdu_adj.pkl --model-name MDN --settings densityForecast --data-module NS --lr 0.00114514 --weight-decay 1.5e-3 --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --gamma 5 --input-dim 25 --output-dim 25 --feature-dim 25 --gradient-clip-val 30 --gpus 1
```

```bash
python3 main.py --data-path data/chengdu_data.pkl --adj-path data/chengdu_adj.pkl --model-name GRU_MDN --settings densityForecast --data-module NS --lr 0.0000114514 --weight-decay 1.5e-3 --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.95 --normalize --gamma 5 --input-dim 25 --output-dim 25 --feature-dim 25 --gradient-clip-val 5 --gpus 1
```

```bash
python3 main.py --data-path data/chengdu_data.pkl --adj-path data/chengdu_adj.pkl --model-name GRU_MDN2 --settings densityForecast --data-module NS --lr 0.0000114514 --weight-decay 1.5e-3 --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.95 --normalize --gamma 5 --input-dim 25 --output-dim 25 --feature-dim 25 --gradient-clip-val 5 --gpus 1
```

Learning rate here must be lower or it will easily gradient exploded.

### Test needed

```bash
python3 main.py --data chengdu --model-name ChebyNet --settings forecast --data-module NS --lr 0.00114514 --weight-decay 1.5e-3 --loss mse --batch-size 1 --seq-len 1 --pre-len 1 --split-ratio 0.8 --normalize --hidden-dim 64 --input-dim 25 --output-dim 25 --feature-dim 25 --gpus 1
```

## monitor metrics

```bash
tensorboard --logdir [LOG DIR PATH]
```

Example:

```bash
tensorboard --logdir ./lightning_logs/GCN
```
