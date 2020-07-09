from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,Model
from glob import glob
import matplotlib.pyplot as plt
import numpy as np


# directory for training images
folders = glob('C:/Users/Mohit/Desktop/Sample Files/Car Dataset/Datasets/Train/*')

resnet = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = Flatten()(resnet.output)
prediction = Dense(len(folders), activation='softmax')(x)


for layer in resnet.layers:
    layer.trainable = False

car_model=Model(inputs=resnet.input,outputs=prediction)
car_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


data_gen=ImageDataGenerator(preprocessing_function=preprocess_input,horizontal_flip=True)

train_data=data_gen.flow_from_directory(r'C:\Users\Mohit\Desktop\Sample Files\Car Dataset\Datasets\Train',batch_size=32,class_mode='categorical',target_size=(224,224))
test_data=data_gen.flow_from_directory(r'C:\Users\Mohit\Desktop\Sample Files\Car Dataset\Datasets\Test',batch_size=32,class_mode='categorical',target_size=(224,224))


History=car_model.fit_generator(train_data,epochs=14,steps_per_epoch=len(train_data),validation_data=test_data,validation_steps=len(test_data))


car_model.save('car_model_resnet50.h5')



y_pred = car_model.predict(test_data)
y_pred = np.argmax(y_pred, axis=1)


plt.plot(History.history['loss'], label='train loss')
plt.plot(History.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')