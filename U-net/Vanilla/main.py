import torch
import numpy as np
import random
from data_loader import Data_Loader
from Vanilla_Trainer import Trainer
from vanilla_loss import VanillaLoss
import sys
sys.path.append('/home/nitin1/segmentation')
from unet_model import UNet
import os


total_epochs = 500
batch_size_is = 4
learning_rate = 0.1
num_classes = 3 ## including background
seeds = [999, 2024]
path_to_images_dir = "../semantic_data/images"
path_to_masks_dir = "../semantic_data/masks"
log_dir = f"../segmentation/Results_segnet/Vanilla_{total_epochs}/Pulp-caries"


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



if __name__ == "__main__":

    for seed in seeds:

        set_seed(seed)

        path_to_log_dir = os.path.join(log_dir, f'seed_{seed}')
        os.makedirs(path_to_log_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        g = torch.Generator()
        g.manual_seed(42)

        dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

        # model = UNet(in_channels=3, out_channels=num_classes, init_features=32)
        model = UNet(in_chn=1, out_chn=num_classes)
        model.to(device)

        with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
            f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\nDevice : {device}\n{"-"*30}\nData Processing : \n{dataset.general_transform}')
        criterion = VanillaLoss(num_class = num_classes)

        trainer = Trainer(model, train_loader, epochs=total_epochs, num_classes=num_classes, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir)

        print("Starting training...")

        trainer.train()
