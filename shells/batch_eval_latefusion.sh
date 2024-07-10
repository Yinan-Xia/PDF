#!/bin/bash
for i in {2..2}
do
  for noise in {0..0}
  do
    for seed in {1..1}
    do
      task="food101"
      task_type="classification"
      model="latefusion"
      name=$task"_"$model"_model_run_$i"
      echo $name noise: $noise seed: $seed
      CUDA_VISIBLE_DEVICES=6 python eval.py --batch_sz 16 --gradient_accumulation_steps 40  \
      --savedir ./saved/$task --name $name  --data_path ./datasets/ \
       --task $task --task_type $task_type  --model $model --num_image_embeds 3 \
       --freeze_txt 5 --freeze_img 3   --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed $seed --df true --noise $noise
    done
  done
done