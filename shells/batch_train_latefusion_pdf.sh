#!/bin/bash
for i in {1..5}
do
  task="food101"
  task_type="classification"
  model="latefusion_pdf"
  batch_sz=32
  lr=5e-05
  weight_decay=1
  name=$task"_"$model"_model_run_df_$i"_"pdf_bz_"$batch_sz"_lr_"$lr"_wd_"$weight_decay
  echo $name
  CUDA_VISIBLE_DEVICES=5 python train_pdf.py --seed $i --weight_decay $weight_decay --batch_sz $batch_sz --gradient_accumulation_steps 1  \
  --savedir ./saved/$task --name $name  --data_path ./datasets/ \
    --task $task --task_type $task_type  --model $model --num_image_embeds 3 \
    --freeze_txt 5 --freeze_img 3   --patience 10 --dropout 0.1 --lr $lr --warmup 0.1 --max_epochs 100 --df true --noise 0
done
# for i in {1..10}
# do
#   task="MVSA_Single"
#   task_type="classification"
#   model="latefusion_pdf"
#   batch_sz=16
#   lr=1e-05
#   weight_decay=1
#   name=$task"_"$model"_model_run_df_$i"_"pdf_bz_"$batch_sz"_lr_"$lr"_wd_"$weight_decay
#   echo $name
#   CUDA_VISIBLE_DEVICES=5 python train_pdf.py --seed $i --weight_decay $weight_decay --batch_sz $batch_sz --gradient_accumulation_steps 1  \
#   --savedir ./saved/$task --name $name  --data_path ./datasets/ \
#     --task $task --task_type $task_type  --model $model --num_image_embeds 3 \
#     --freeze_txt 5 --freeze_img 3   --patience 10 --dropout 0.1 --lr $lr --warmup 0.1 --max_epochs 100 --df true --noise 0
# done
