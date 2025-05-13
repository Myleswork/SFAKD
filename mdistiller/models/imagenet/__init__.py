from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnet18_test, resnet34_test, resnet50_test, resnet101_test, resnet152_test
from .mobilenetv2 import MobileNetV2,MobileNetV2_test


imagenet_model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50":{
        0:resnet50,
        1:"/home/zhao/project/CAT-KD/download_ckpts/imagenet_teacher/imagenet_r50_76.358",
    },
    "ResNet101": resnet101,
    "MobileNetV2": MobileNetV2,
    "ResNet18_test": resnet18_test,
    "ResNet34_test": resnet34_test,
    "ResNet50_test": resnet50_test,
    "ResNet101_test": resnet101_test,
    "MobileNetV2_test": MobileNetV2_test,
}
