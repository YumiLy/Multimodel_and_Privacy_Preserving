#!/bin/bash
#SBATCH --partition=gpu-1jour
#SBATCH --job-name=speaker_identification
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs_BS.out
#SBATCH --error=results/logs_BS.err
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



====== Launch job for different batch sizes ======
for bs in 1 2 4 8; do
    echo "=============================="
    echo "Running with batch size: $bs"
    echo "=============================="
    python /home/lai/models/Wav2Vec2/ASR_BeamSearch.py --batch_size_emb=$bs --trim_test 0
done



