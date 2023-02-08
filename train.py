import wandb
import json
import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_squared_error
from utils.datasets import Vision_Lags2One, get_df
from utils.models import KIFNet
from utils import EarlyStopping, GaussianNoise
from utils import split_rmse

wandb.init()

# cuda
cuda_N = 1
device = f'cuda:{cuda_N}'
n_gpu = 1

# config with features and files
CONFIG_FILE = './configs/left_leg_config_all_lower.json'
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

wandb.log_artifact(CONFIG_FILE, name='config_file', type='split_config')

# main wandb config
wandb.config.update({
    'dataset': dict(
        lags=list(range(46)),
        ds_mean=[0.32366217, 0.33821055, 0.21849849],
        ds_std=[0.15844604, 0.16612123, 0.12687401],
        resize=(224, 224),
        config=config,
        batch_size = 256 * n_gpu,
        n_gpu=n_gpu,
        train_stride=3,
        valid_stride=1,
        offset=29,
        num_workers=32,
    ),
    'model': dict(
        inp_dim=184,
        out_dim=2,
        hidden_dims=[256,256,128],
        kinematic_emb=128,
        fusion_dims=[64,32],
        image_emb=128,
        mobone_type='s0',
        mobone_path = './ml-mobileone/weights/mobileone_s0_unfused.pth.tar',
        fusion='adain-cv-kin',
        device=device
    ),
    'training': dict(
        early_stopping_rounds=5,
        learning_rate=1e-4,
        loss='mse',
        optimizer='adam',
        gaussian_noise=0.015,
        max_epochs=30
    ),
    'testing': dict(
        split_window=7*30,
        alpha_ankle=0.9
    )
})

# train/val/test datasets
train_dataset = ConcatDataset([
    Vision_Lags2One(
        data=get_df(file),
        lags=wandb.config.dataset['lags'],
        offset=wandb.config.dataset['offset'],
        stride=wandb.config.dataset['train_stride'],
        resize=wandb.config.dataset['resize'],
        ds_mean=wandb.config.dataset['ds_mean'],
        ds_std=wandb.config.dataset['ds_std'],
        config=wandb.config.dataset['config'],
    ) for file in tqdm(config['train_files'])
])

train_dataloader = DataLoader(train_dataset, batch_size=wandb.config.dataset['batch_size'],
                              shuffle=True,
                              num_workers=wandb.config.dataset['num_workers'])
print("Number of batches in train dataset:", len(train_dataset))

valid_dataset = ConcatDataset([
    Vision_Lags2One(
        data=get_df(file),
        lags=wandb.config.dataset['lags'],
        offset=wandb.config.dataset['offset'],
        stride=wandb.config.dataset['valid_stride'],
        resize=wandb.config.dataset['resize'],
        ds_mean=wandb.config.dataset['ds_mean'],
        ds_std=wandb.config.dataset['ds_std'],
        config=wandb.config.dataset['config'],
    ) for file in tqdm(config['valid_files'])
])

valid_dataloader = DataLoader(valid_dataset, batch_size=wandb.config.dataset['batch_size'],
                              num_workers=wandb.config.dataset['num_workers'])

print("Number of batches in valid dataset:", len(valid_dataset))


test_dataset = ConcatDataset([
    Vision_Lags2One(
        data=get_df(file),
        lags=wandb.config.dataset['lags'],
        offset=wandb.config.dataset['offset'],
        stride=wandb.config.dataset['valid_stride'],
        resize=wandb.config.dataset['resize'],
        ds_mean=wandb.config.dataset['ds_mean'],
        ds_std=wandb.config.dataset['ds_std'],
        config=wandb.config.dataset['config'],
    ) for file in tqdm(config['valid_files'])
])

test_dataloader = DataLoader(test_dataset, batch_size=wandb.config.dataset['batch_size'],
                             num_workers=wandb.config.dataset['num_workers'])

print("Number of batches in test dataset:", len(test_dataset))

# model initialization
model = KIFNet(**wandb.config.model).to(device)
model = nn.DataParallel(model, device_ids=[cuda_N]).to(device)

early_stopping = EarlyStopping(patience=wandb.config.training['early_stopping_rounds'])

optimizer = torch.optim.Adam(model.parameters(),
    lr=wandb.config.training['learning_rate']
)

noise = GaussianNoise(sigma=wandb.config.training['gaussian_noise'])

mse_loss = torch.nn.MSELoss()
scaler = GradScaler()

all_params = sum(p.numel() for p in model.parameters())
grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable params:{grad_params}/{all_params} ~ {100*(grad_params/all_params):.2f}%")
 
def test(dataloader, model):
    model.eval()
    pbar_val = tqdm(dataloader)
    preds, y_true = [],[]
    for i, (kinematic_val, y_val, cv_val) in enumerate(pbar_val, start=1):
        kinematic_val = kinematic_val.to(device)
        cv_val = cv_val.to(device)
        
        with autocast():
            out_val = model(kinematic_val, cv_val)
        
        preds.append(out_val.detach().cpu().numpy())
        y_true.append(y_val.numpy())
    
    true_vals = np.vstack(y_true)
    pred_vals = np.vstack(preds)
    
    ankle_mean_val, _ = split_rmse(
        torch.Tensor(pred_vals[:, 0]), torch.Tensor(true_vals[:, 0]),
        wandb.config.testing['split_window'])
    
    knee_mean_val, _ = split_rmse(
        torch.Tensor(pred_vals[:, 1]), torch.Tensor(true_vals[:, 1]),
        wandb.config.testing['split_window'])
    
    ankle_val = mean_squared_error(true_vals[:,0], pred_vals[:,0], squared = False)
    knee_val = mean_squared_error(true_vals[:,1], pred_vals[:,1], squared = False)
    
    return ankle_val, ankle_mean_val, knee_val, knee_mean_val

# train/validation loop
weight_losses = torch.Tensor([wandb.config.testing['alpha_ankle'], 1-wandb.config.testing['alpha_ankle']]).to(device)

for epoch in range(wandb.config.training['max_epochs']):
    model.train()
    pbar = tqdm(train_dataloader)
    total_loss = 0
    for i, (X, y, cv) in enumerate(pbar, start=1):
        
        X = noise(X).to(device)
        y = y.to(device)
        cv = cv.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            out = model(X, cv)
            loss = weight_losses.dot(torch.sqrt(((out-y)**2).mean(0)))
        
        total_loss += loss.item()
        pbar.set_description(f'Average batch RMSE: {total_loss / i:.4f}', refresh=True)
        
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        
    test_ankle, test_ankle_mean, test_knee, test_knee_mean = test(test_dataloader, model)
    _, valid_ankle_mean, _, valid_knee_mean = test(valid_dataloader, model)
    
    wandb.log({'train_loss': (total_loss / i),
               'valid_knee_mean': valid_knee_mean,
               'valid_ankle_mean': valid_ankle_mean})
    
    print(f'Test epoch={epoch}, Overall Ankle RMSE={test_ankle:.4f}, mean={test_ankle_mean:.4f}')
    print(f'Test epoch={epoch}, Overall Knee RMSE={test_knee:.4f}, mean={test_knee_mean:.4f}')
    wandb.log({'test_ankle': test_ankle, 'test_knee_mean': test_knee_mean,
               'test_ankle_mean': test_ankle_mean, 'test_knee': test_knee})
    
    early_stopping(valid_knee_mean + valid_ankle_mean, model)
    if early_stopping.early_stop:
        print('Early stopping!')
        break   
        
model.load_state_dict(early_stopping.best_weights)

# testing the model
test_ankle, test_ankle_mean, test_knee, test_knee_mean = test(test_dataloader, model)
_, valid_ankle_mean, _, valid_knee_mean = test(valid_dataloader, model)

wandb.log({'valid_knee_mean': valid_knee_mean, 'valid_ankle_mean': valid_ankle_mean})

print(f'Overall Test Ankle RMSE={test_ankle:.4f}, mean={test_ankle_mean:.4f}')
print(f'Overall Test Knee RMSE={test_knee:.4f}, mean={test_knee_mean:.4f}')
wandb.log({'test_ankle': test_ankle, 'test_knee_mean': test_knee_mean,
            'test_ankle_mean': test_ankle_mean, 'test_knee': test_knee})

# saving model
torch.save(early_stopping.best_weights, 'trained_model.pt')
