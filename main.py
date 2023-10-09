import torch
import torchvision
import datetime
from torchvision import transforms as T

from ss_train import ss_train


transforms = {
    'train': T.Compose([
        T.ToPILImage(),
        # T.RandomResizedCrop(3000),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ]),
    'valid': T.Compose([
        T.ToPILImage(),
        #  T.CenterCrop(3000),
        #  T.Resize(500),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for lr in [0.01]:
        # model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        model = torchvision.models.resnet18(weights='DEFAULT')
        final_layer_in = model.fc.in_features
        model.fc = torch.nn.Linear(final_layer_in, 2)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # optimizer = torch.optim.Adam(model.parameters())

        # model, best_ep = ss_train(
        #     model, criterion, optimizer, 200, start_alpha_from=10, reach_max_alpha_in=150)

        model, best_ep = ss_train(model, criterion, optimizer, device, 64, 200, start_alpha_from=10, reach_max_alpha_in=150,
                                  max_alpha=1, scheduler=None, base_data_dir="/content/FloodNet/", transforms=transforms, train_size=0.8)
        current_time = datetime.datetime.now()

        torch.save({
            'epoch': best_ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"/content/checkpoints/fn_{lr}_{best_ep}_{current_time.day}_{current_time.hour}_{current_time.minute}_best.pt")

        return (model, best_ep)


if __name__ == '__main__':
    main()
