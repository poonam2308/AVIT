unset WANDB_MODE
wandb login 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
WANDB_MODE=online python main.py --config configs/imagenette.yaml
#
#WANDB_MODE=online python main.py --config configs/imagewoof.yaml

#WANDB_MODE=online python main.py --config configs/cifar10.yaml