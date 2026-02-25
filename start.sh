#unset WANDB_MODE
#WANDB_MODE=online python main.py --config configs/imagenette.yaml
#python main.py --config configs/imagewoof.yaml

unset WANDB_MODE
WANDB_MODE=online python main.py --config configs/cifar10.yaml