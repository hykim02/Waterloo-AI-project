from pycocotools.coco import COCO
import torchvision
import torch
import torch.nn as nn
from torchvision.ops import *
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

# 모델을 GPU에서 실행하기 위해 cuda로 설정
device = torch.device('cuda')

from functions import *

# 데이터 셋 설정
testing_dataset = COCODataset(image_dir='./images', json_path='./test.json', category_ids=categories, device=device)

# 입력데이터를 원하는 형식으로 변경하는 함수
def collate_fn(batch):
    return tuple(zip(*batch))

# SSD-Lite 모델을 MobileNet V3 Large 백본으로 사용
# 사전 학습된 가중치로 사용
# 클래스 수 
# 학습 가능한 레이어 수 = 6
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2, num_classes=len(categories) + 1, trainable_backbone_layers=6)

# 학습 가능한 파라미터 추출
# 최적화 함수 : Adam
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 횟수 1000회
num_epochs = 1000

# 모델이 사용할 디바이스 채택
model.to(device)

# 케이크 조각 크기 설정 
batch_size=60

# 에포크 수만큼 반복 학습
for epoch in range(num_epochs):
    model.train()
    training_dataset = COCODataset(image_dir='./images', json_path='./train.json', category_ids=categories, device=device)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    i = 0
    total_loss = 0

    for images, targets in training_dataloader:
        i += 1

        # move the images and targets to the device
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss = total_loss + (losses/batch_size)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # print average loss
        print(f"Epoch: {epoch}, Loss: {total_loss/i}")

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        # metric = testing_dataset.coco_evaluate(model, testing_dataloader, device)
        # print(f"Epoch: {epoch}, mAP: {metric}")
        # save the model with epoch number
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")


for images, targets in training_dataloader:
    prediction = model(images)
    print('HERE')

    boxes = prediction[0]['boxes']
    boxes = boxes.cpu().detach().numpy()

    boxes = [boxes[prediction[0]['scores'].argmax()]]
    plot_output_image_with_annotations[images[0], boxes]

    class_weights = [0, 0, 0, 0, 0]
    for i in range(len(prediction[0]['scores'])):
        class_weights[prediction[0]['labels'][i]] = class_weights[prediction[0]['labels'][i]] + prediction[0]['scores'][i]

    print(class_weights)

print('HERE')