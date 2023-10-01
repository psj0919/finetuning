import reference.transforms as T
import torch

from reference.engine import train_one_epoch, evaluate
import utils
from dataset.torchvision_dataset import PennFudanDataset
from model.torchvision_model import get_model_instance_segmentation
from torchvision.transforms import ConvertImageDtype



def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__=='__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    gpu_id = "0"
    device = torch.device("cuda:{}" .format(gpu_id) if torch.cuda.is_available() else "cpu")

    # -------------download_engine,utils,coco_utils,coco_eval,transforms------------------------------------
    # os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
    # os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
    # os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
    # os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
    # os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")
    # ---------------------------------------------------------------------------------------------------------

    # our dataset has two classes only - background and person
    num_classes = 2
    # ---------------------------------------------------------
    # use our dataset and defined transformations
    dataset = PennFudanDataset('/storage/sjpark/PennFudanPed/', get_transform(train=True))
    dataset_test = PennFudanDataset('/storage/sjpark/PennFudanPed/', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test datasetutils
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
