from torchinfo import summary

from models.detection.rotated_faster_rcnn import rotated_fasterrcnn_resnet50_fpn_v2

net = rotated_fasterrcnn_resnet50_fpn_v2(pretrained=True, progress=True, num_classes=13, pretrained_backbone=True)
net.train()
net.cuda()

# TODO: input_data with real data
summary(net)