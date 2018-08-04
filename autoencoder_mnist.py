import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
import matplotlib.pyplot as plt


x_img, y_img = 28, 28
input_shape = (x_img,y_img,1)
input_layer = Input(input_shape)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  


x = Conv2D(16, 5, activation='relu')(input_layer)
x = MaxPooling2D(2)(x)
x = Conv2D(20, 2, activation='relu')(x)
x = MaxPooling2D(2)(x)
encoded = x
x = UpSampling2D(2)(x)
x = Conv2DTranspose(20, 2, activation='relu')(x)
x = UpSampling2D(2)(x)
x = Conv2DTranspose(10, 5, activation='relu')(x)
x = Conv2DTranspose(1, 3, activation='sigmoid')(x)
decoded = x

autoencoder = Model(input_layer, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True, 
				validation_data=(x_test, x_test))
autoencoder.save("autoencoder.h5")

#To load model and test
#autoencoder = load_model('autoencoder.h5')

test_img = x_test[10]
test_img = np.reshape(test_img, (1, 28, 28, 1)) 
test_prediction = autoencoder.predict(test_img)

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(test_img.reshape(28, 28))
plt.subplot(1, 2, 2)
plt.title('Decoded')
plt.imshow(test_prediction.reshape(28, 28))
plt.gray()


