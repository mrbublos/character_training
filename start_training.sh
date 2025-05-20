#!/bin/bash
# python /app/character_training/otniel_scripts/caption_with_florence-2.py /dataset --output_dir /train_dataset
accelerate launch --num_processes 1 --mixed_precision bf16 --num_cpu_threads_per_process 2 ./run.py $1