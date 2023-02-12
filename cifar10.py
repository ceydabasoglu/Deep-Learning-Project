
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import seaborn as sns


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print(x_train.shape)
print(x_test.shape)

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def plot_sample(x, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[np.argmax(y[index])])
    

   
    
# Normalize data
def pixels(train, test):
    x_train = train.astype('float32')
    x_test =  test.astype('float32') 
    
    x_train = x_train / 255.
    x_test = x_test / 255.
    return x_train,x_test


y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Evaluate
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss : ' ,scores[0])
print('Test accuracy:', scores[1])
from sklearn.metrics import classification_report
y_pred = model.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
from keras.utils import to_categorical
y_pred_onehot = to_categorical(y_pred_classes)
print("Classification Report: \n", classification_report(y_test, y_pred_onehot))


y_pred = model.predict(x_test)
print(y_pred[:5])
y_classes = [np.argmax(element) for element in y_pred]
print(y_classes[:5])
print(y_test[:5])
plot_sample(x_test, y_test, 3) #airplane
print(classes[y_classes[3]]) 

fig, axs = plt.subplots(1,2, figsize = (15,5))
sns.countplot(y_train.ravel(), ax=axs[0])
axs[0].set_title('Distribution of training data')
axs[0].set_xlabel('classes')

sns.countplot(y_test.ravel(), ax = axs[1])
axs[1].set_title('Distribution of testing data')
axs[1].set_xlabel('classes')
plt.show()


fig, axs = plt.subplots(1,2, figsize = (15,5))
axs[0].plot(history.history['accuracy'])
axs[0].plot(history.history['val_accuracy'])
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['train','validate'], loc='upper left')

axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['train','validate'], loc='upper left')
plt.show()


fix, axes = plt.subplots(5,5, figsize=(12,12))
axes = axes.ravel()

for i in np.arange(0,25):
    axes[i].imshow(x_test[i])
    axes[i].set_title("True: %s \nPredict: %s" % (classes[y_classes[i]], classes[y_pred_classes[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)
    


misclassified_idx = np.where(y_pred_classes != y_classes)[0]
if len(misclassified_idx) > 0:
    fix, axes = plt.subplots(3,5, figsize = (12,8))
    axes = axes.ravel()
    for i in np.arange(0,15):
        axes[i].imshow(x_test[misclassified_idx[i]])
        axes[i].set_title("True: %s \nPredict: %s" % (classes[y_classes[misclassified_idx[i]]], classes[y_pred_classes[misclassified_idx[i]]]))
        axes[i].axis('off')
        plt.subplots_adjust(wspace=1)
else:
    print("There are no misclassified examples.")


def show_prediction(number):
    fig, ax = plt.subplots(figsize=(3, 3))
    test_image = np.expand_dims(x_test[number], axis=0)
    prediction = model.predict_classes(test_image)[0]
    ax.imshow(x_test[number])
    true_label = np.argmax(y_test[number])
    ax.set_title("Predicted: {} \nTrue Label: {}".format(classes[prediction], classes[true_label]))
    plt.axis('off')
    plt.show()
show_prediction(10)
#save model 

model.save('cifar10_train_model.h5')
from keras.models import load_model
loaded_model = load_model('cifar10_train_model.h5')
loaded_model.summary()