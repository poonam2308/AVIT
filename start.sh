wandb login <YOUR_API_KEY>
python main.py --config configs/imagenette.yaml \
  --override gating.target_keep_ratio=0.5