import torch
from tqdm.auto import tqdm
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.utils.data import DataLoader

from dataset.utils import labeled_ds, unlabeled_ds
from dataset.dataset import FloodNetTsk1


def create_dataloaders(base_data_dir='/content/FloodNet/', transforms=None, train_size=0.8):
    X_train, X_valid, y_train, y_valid = labeled_ds(
        base_data_dir, train_size=train_size)
    unlabeled_file_names = unlabeled_ds(base_data_dir)

    unique_labels, counts = np.unique(y_train, return_counts=True)
    class_weights = [1/c for c in counts]
    # class_weights[1]*=1.5 # SKEWING THE SAMPLER MORE
    sample_weights = [0] * len(y_train)

    for idx, lbl in enumerate(y_train):
        sample_weights[idx] = class_weights[lbl]

    sampler_train = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    pseudo_labels = np.array([-1]*len(unlabeled_file_names))

    train_set = FloodNetTsk1(X_train, y_train, transform=transforms['train'])
    valid_set = FloodNetTsk1(X_valid, y_valid, transform=transforms['valid'])
    unlabeled_set = FloodNetTsk1(
        unlabeled_file_names, None, transform=transforms['valid'])

    return train_set, valid_set, unlabeled_set, pseudo_labels, sampler_train


def ss_train(model, criterion, optimizer, device, BATCH_SIZE, num_epochs, start_alpha_from=5, reach_max_alpha_in=15, max_alpha=1,
             scheduler=None, base_data_dir='/content/FloodNet/', transforms=None, train_size=0.8):

    train_set, valid_set, unlabeled_set, pseudo_labels, sampler_train = create_dataloaders(
        base_data_dir, transforms=transforms, train_size=train_size)

    dataset_size = {
        'train': len(train_set),
        'valid': len(valid_set)
    }

    dataloaders = {
        'train': DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler_train),
        'valid': DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True),
        'unlabeled': DataLoader(unlabeled_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    }

    writer = SummaryWriter()
    alphas = np.linspace(0, max_alpha, reach_max_alpha_in-start_alpha_from)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(num_epochs):

        print('-'*80)
        if epoch < start_alpha_from:
            alpha = 0
        elif epoch-start_alpha_from >= len(alphas):
            alpha = alphas[-1]
        else:
            # alpha = ((start_alpha_from-max_alpha)/(reach_max_alpha_in-0))*epoch
            # alpha = ((max_alpha-0)/(reach_max_alpha_in-start_alpha_from))*epoch # Correct linearly alpha ramping equation
            alpha = alphas[max(0, epoch-start_alpha_from)]

        for phase in ['train', 'unlabeled', 'valid']:
            print(phase)
            print('-'*5)
            if alpha == 0 and phase == 'unlabeled':
                continue
            if phase in ['train', 'unlabeled']:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # for confusion matrix
            epoch_preds = []
            epoch_lbls = []

            for batch_id, (img, lbl) in enumerate(tqdm(dataloaders[phase], desc=f"Epoch {epoch}/{num_epochs}, {phase}, alpha:{alpha:.2f}")):
                img = img.to(device)
                if phase in ['train', 'valid']:
                    lbl = lbl.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase in ['train', 'unlabeled']):
                    output = model(img)
                    preds = torch.argmax(output, 1)

                    if phase in ['train', 'valid']:
                        loss = criterion(output, lbl)
                    else:
                        loss = alpha * \
                            criterion(output, torch.tensor(
                                pseudo_labels[lbl], dtype=torch.int64).to(device))

                    if phase in ['train', 'valid']:
                        epoch_preds.extend(preds.detach().cpu())
                        epoch_lbls.extend(lbl.detach().cpu())

                    if phase in ['train', 'unlabeled']:
                        loss.backward()
                        optimizer.step()

                if phase in ['train', 'valid']:
                    writer.add_scalars(f'Step-Loss/{phase}', {'loss': loss.item(
                    ), 'alpha': alpha}, (len(dataloaders[phase])*epoch)+batch_id)
                    running_loss += loss.item() * img.size(0)
                    running_corrects += torch.sum(preds == lbl.data)

            # if phase in ['train', 'valid']:
            #   print(confusion_matrix(epoch_lbls, epoch_preds), np.array([0, 1]))

            if scheduler is not None and phase == 'train':
                scheduler.step()

            if phase in ['train', 'valid']:
                epoch_f1 = f1_score(epoch_lbls, epoch_preds)
                epoch_loss = running_loss / dataset_size[phase]
                epoch_acc = running_corrects.double() / dataset_size[phase]

            if phase in ['train', 'valid']:
                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                writer.add_scalars(
                    f'F1/{phase}', {'f1': epoch_f1, 'alpha': alpha}, epoch)
                writer.add_scalars(
                    f'Accuracy/{phase}', {'accu': epoch_acc, 'alpha': alpha}, epoch)

                print(
                    f'Epoch {epoch}, {phase} Loss: {epoch_loss:.4f}, Acc:{epoch_acc:.4f}, F1:{epoch_f1:.4f}')

            if phase == 'train' and epoch >= start_alpha_from-1:
                model.eval()
                for img, lbl in tqdm(dataloaders['unlabeled'], desc="Predicting pseudo labels"):
                    img = img.to(device)
                    preds = torch.argmax(model(img), 1)
                    pseudo_labels[lbl] = preds.detach().cpu()

            if phase == 'valid' and epoch_f1 > best_f1:
                best_epoch = epoch
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"/content/checkpoints/best_ckpt.pt")

    time_elapsed = time.time() - since
    writer.close()
    current_time = datetime.datetime.now()

    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"/content/checkpoints/fn_{num_epochs}_{current_time.day}_{current_time.hour}_{current_time.minute}_not_best.pt")

    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Saved model is from {best_epoch} epoch with f1 {best_f1}')

    model.load_state_dict(best_model_wts)

    return (model, best_epoch)
