import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from vision.common_utils import checkpoint

def generate_image(history, image_name, output_dir):
    labels = tuple(history.keys())
    epoch_num = len(history['train'])

    x = np.arange(epoch_num)
    train_y = np.array(history['train'])
    val_y = np.array(history['val'])

    plots = plt.plot(x, train_y, label='train')
    plots = plt.plot(x, val_y, label='val')
    plt.legend(loc='best')
    plt.xlabel("epoch")
    plt.savefig(os.path.join(output_dir, image_name+".png"))
    plt.clf()

def do_train(cfg, model, data_loader, data_loader_val,
             optimizer,  criterion,   device):
    model = model.to(device)
    if device == 'cuda:0':
        model = model.to(device)
        torch.backends.cudnn.benchmark = True

    checkpointer = checkpoint.CheckPointer(cfg, model, optimizer, criterion)

    dataloader_dict = {'train' : data_loader, 'val' : data_loader_val}
    loss_history = {'train': [], 'val': []}
    acc_history  = {'train': [], 'val': []}
    checkpointed_epoch = checkpointer.load()

    for epoch in range(checkpointed_epoch, cfg.SOLVER.MAX_EPOCH):
        print('Epoch {}/{}'.format(epoch+1, cfg.SOLVER.MAX_EPOCH))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc), flush=True)
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(float(epoch_acc))
        checkpointer.save("vision", epoch)

    save_path = os.path.join(cfg.OUTPUT_DIR, "vision.pth")
    torch.save(model.state_dict(), save_path)
    checkpointer.remove()
    generate_image(loss_history, "Loss", cfg.OUTPUT_DIR)
    generate_image(acc_history, "Accuracy", cfg.OUTPUT_DIR)
