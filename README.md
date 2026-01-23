# D2R

## Environment Requirements

The experiments were conducted on a single GeForce RTX 3090 GPU with 24GB memory. 

* Python 3.7.16
* PyTorch 1.7.1
* CUDA 11.2

## Run Code

### runS.py

```shell
python runS.py --num_epochs 20 --lr 1e-5 --warmup_ratio 0.2 --seed 3407 --batch_size 64 --max_seq 64 --weight_js_1 0.9 --weight_js_2 0.3 --DR_step 4 --weight_diff 0 --device cuda:1
```
```shell
--num_epochs 20 --lr 1e-5 --warmup_ratio 0.2 --seed 3407 --batch_size 64 --max_seq 64 --weight_js_1 0.9 --weight_js_2 0.3 --DR_step 4 --weight_diff 0 --device cuda:1
```

### runM.py

```shell
python runM.py --num_epochs 20 --lr 1e-5 --warmup_ratio 0.2 --seed 3407 --batch_size 64 --max_seq 64 --weight_js_1 0.9 --weight_js_2 0.3 --DR_step 4 --weight_diff 0 --device cuda:1
```
```shell
--num_epochs 20 --lr 1e-5 --warmup_ratio 0.2 --seed 3407 --batch_size 64 --max_seq 64 --weight_js_1 0.9 --weight_js_2 0.3 --DR_step 4 --weight_diff 0 --device cuda:1
```

### runH.py

```shell
python runH.py --num_epochs 20 --lr 2e-5 --warmup_ratio 0.2 --seed 3407 --batch_size 64 --max_seq 64 --weight_js_1 0.6 --weight_js_2 1.0 --DR_step 3 --weight_diff 0 --device cuda:1
```
```shell
--num_epochs 20 --lr 2e-5 --warmup_ratio 0.2 --seed 3407 --batch_size 64 --max_seq 64 --weight_js_1 0.6 --weight_js_2 1.0 --DR_step 3 --weight_diff 0 --device cuda:1
```

🐮🐴