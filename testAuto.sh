source ~/miniconda3/etc/profile.d/conda.sh
conda activate audioldm_train
poetry install

python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2025_01_31_test_audioldm/audioldm_tinytest.yaml --reload_from_ckpt data/checkpoints/audioldm-m-full.ckpt