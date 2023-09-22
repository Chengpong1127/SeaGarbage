from GarbageDataset import GarbageDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, YolosModel, YolosForObjectDetection, YolosConfig
import torch
import argparse
import yaml
import os

def label_to_device(labels, device):
    for label in labels:
        for key in label:
            label[key] = label[key].to(device)
    return labels

def save_checkpoint(model, epochs, filename):
    checkpoint = {
        'model': model.state_dict(),
        'epochs': epochs
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint['epochs']


def main(config):
    dataset = GarbageDataset(config['model_name'], config['label_path'], config['dataset_dir'], config['image_size'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    model = YolosForObjectDetection(YolosConfig.from_pretrained(config['model_name']))
    

    # Load Checkpoint
    saved_epoch = 0
    if os.path.exists(config['checkpoint_dir']):
        checkpoint = os.listdir(config['checkpoint_dir'])
        if len(checkpoint) > 0:
            checkpoint.sort()
            model, saved_epoch = load_checkpoint(model, os.path.join(config['checkpoint_dir'], checkpoint[-1]))
            print(f'Load Checkpoint: {checkpoint[-1]}')
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    model.to(config['device'])

    for epoch in range(saved_epoch, config['epochs']):
        save_filename = f'epoch_{epoch}.pth'
        save_checkpoint(model, epoch, os.path.join(config['checkpoint_dir'], save_filename))
        log_loss = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(config['device'])
            labels = label_to_device(labels, config['device'])
            outputs = model(inputs, labels = labels)
            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            log_loss += loss.item()
        print(f'Epoch: {epoch}, Loss: {log_loss/len(dataloader)}')
        
    
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str)
    args = argparser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    main(config)