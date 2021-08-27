from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
# look up the structures of DenseNet, EfficientNet, ResNet
model = VGG16()
model = VGG19()
model = Xception()
model = ResNet50()
model = ResNet50V2()
model = ResNet101()
model = ResNet101V2()
model = ResNet152()
model = ResNet152V2()
model = DenseNet121()
model = DenseNet169()
model = DenseNet201()
model = InceptionV3()
model = InceptionResNetV2()
model = MobileNet()
model = MobileNetV2()
model = MobileNetV3Large()
model = MobileNetV3Small()
model = NASNetLarge()
model = NASNetMobile()
model = EfficientNetB0()
model = EfficientNetB1()
model = EfficientNetB7()


model.trainable=False
model.summary()
print('========================  ========================')
print('the number of parameters:',)
print('the number of trainable parameters:',)
print('the number of Weights:',len(model.weights))
print('the number of trainable Weights:',len(model.non_trainable_weights))
