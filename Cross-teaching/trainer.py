import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

def train(unet_model,
          swin_model,
          train_loader, 
          val_loader, 
          unlabeled_loader,
          unet_criterion, 
          unet_optimizer, 
          unet_scheduler,
          swin_criterion,
          swin_optimizer,
          swin_scheduler, 
          device, 
          patience=3, 
          num_epochs=20, 
          save_path='best_model.pt'):
    
    unet_model.to(device)
    swin_model.to(device)

    it = 0

    unet_best_loss = float('inf')
    swin_best_loss = float('inf')

    best_epoch = 0
    no_improvement = 0

    unet_train_losses = []
    unet_val_losses = []

    swin_train_losses = []
    swin_val_losses = []

    fig, ax = plt.subplots()  # Create a figure and axis object for the plot

    for epoch in range(1, num_epochs + 1):
        unet_model.train()
        swin_model.train()

        unet_total_loss = 0
        swin_total_loss = 0

        print('** '*40)
        print('Epoch [{}/{}]'.format(epoch, num_epochs))
        print('Training')

        if (it % 2) == 0:
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(device), target.to(device)
                data = data[None, :]
                target = target.type(torch.LongTensor).to(device)

                # unet
                unet_optimizer.zero_grad()
                unet_output = unet_model(data)
                unet_loss = unet_criterion(unet_output, target)
                unet_loss.backward()
                unet_optimizer.step()
                unet_total_loss += unet_loss.item()

                # swin unet
                swin_optimizer.zero_grad()
                swin_output = swin_model(data)
                swin_loss = swin_criterion(swin_output, target)
                swin_loss.backward()
                swin_optimizer.step()
                swin_total_loss += swin_loss.item()
            
            # losses
            unet_avg_loss = unet_total_loss / len(train_loader)
            swin_avg_loss = swin_total_loss / len(train_loader)
        else:
            for batch_idx, data in enumerate(tqdm(unlabeled_loader)):
                data, target = data.to(device), target.to(device)
                data = data[None, :]
                target = target.type(torch.LongTensor).to(device)

                # unet
                unet_optimizer.zero_grad()
                unet_output = unet_model(data)

                # swin unet
                swin_optimizer.zero_grad()
                swin_output = swin_model(data)

                # cross teaching unet
                unet_loss = unet_criterion(unet_output, swin_output)
                unet_loss.backward()
                unet_optimizer.step()
                unet_total_loss += unet_loss.item()

                # cross teachin swin
                swin_loss = swin_criterion(swin_output, unet_output)
                swin_loss.backward()
                swin_optimizer.step()
                swin_total_loss += swin_loss.item()

            # losses
            unet_avg_loss = unet_total_loss / len(unlabeled_loader)
            swin_avg_loss = swin_total_loss / len(unlabeled_loader) 

        it += 1
        
        # unet loss
        unet_train_losses.append(unet_avg_loss)
        print('Epoch [{}/{}], Unet Train Loss: {:.4f}'.format(epoch, num_epochs, unet_avg_loss))
        print('Validation')
        unet_val_loss, unet_val_accuracy = test(unet_model, val_loader, unet_criterion, device)
        unet_val_losses.append(unet_val_loss)
        print('Epoch [{}/{}], Unet Validation Loss: {:.4f}, Unet Validation Accuracy: {:.2f}%'.format(epoch, num_epochs, unet_val_loss, unet_val_accuracy))

        # swin loss
        swin_train_losses.append(swin_avg_loss)
        print('Epoch [{}/{}], Unet Train Loss: {:.4f}'.format(epoch, num_epochs, swin_avg_loss))
        print('Validation')
        swin_val_loss, swin_val_accuracy = test(swin_model, val_loader, swin_criterion, device)
        swin_val_losses.append(swin_val_loss)
        print('Epoch [{}/{}], Unet Validation Loss: {:.4f}, Unet Validation Accuracy: {:.2f}%'.format(epoch, num_epochs, swin_val_loss, swin_val_accuracy))

        unet_scheduler.step(unet_val_loss)
        swin_scheduler.step(swin_val_loss)

        if unet_val_loss < unet_best_loss or swin_val_loss < swin_best_loss:
            best_epoch = epoch
            torch.save(unet_model.state_dict(), save_path)
            no_improvement = 0
        elif unet_val_loss < unet_best_loss:
            unet_best_loss = unet_val_loss
        elif swin_val_loss < swin_best_loss:
            swin_best_loss = swin_val_loss
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print('Early stopping. No improvement in validation loss for {} epochs.'.format(no_improvement))
                break

        # Update the plot after each epoch
        ax.plot(range(1, epoch + 1), unet_train_losses, label='Unet Training Loss')
        ax.plot(range(1, epoch + 1), unet_val_losses, label='Unet Validation Loss')
        ax.plot(range(1, epoch + 1), swin_train_losses, label='Swin Training Loss')
        ax.plot(range(1, epoch + 1), swin_val_losses, label='Swin Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        plt.savefig('loss_plot.png')  # Save the plot as an image

    print('Best model achieved at epoch {}'.format(best_epoch))

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            data = data[None, :]
            target = target.type(torch.LongTensor).to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = torch.argmax(output, dim=1)
            iou = compute_iou(pred, target)
            total_iou += iou.item()

            total_samples += data.size(0)

    avg_loss = total_loss / len(test_loader)
    avg_iou = total_iou / total_samples
    return avg_loss, avg_iou

def compute_iou(pred, target):
    intersection = torch.logical_and(pred, target).sum((1, 2, 3))
    union = torch.logical_or(pred, target).sum((1, 2, 3))
    iou = (intersection.float() + 1e-6) / (union.float() + 1e-6)
    return iou.mean()
