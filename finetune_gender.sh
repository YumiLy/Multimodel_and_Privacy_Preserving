#!/bin/bash
#SBATCH --partition=gpu-1semaine
#SBATCH --job-name=finetune_gender
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6-20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lai@isir.upmc.fr


# ====== Load your environment ======
source ~/anaconda3/etc/profile.d/conda.sh
conda activate multimodal

# ====== Optional: sanity check ======
echo "Running on host: $(hostname)"
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0  # Disable for better throughput


for idx in 6 8 10 ; do
    echo "Running with freeze_layers: $idx"
    date  # 打印当前时间
    echo "=============================="
    echo "Running with freeze_layers: $idx"
    echo "=============================="
    python /home/lai/models/Wav2Vec2/finetune_gender.py --freeze_layers=$idx \
        > results/log_finetune_gender_layer${idx}.out \
        2> results/log_finetune_gender_layer${idx}.err
    date
done