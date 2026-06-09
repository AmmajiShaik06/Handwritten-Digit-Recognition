#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.makedirs("model", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
os.makedirs("report", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

print("Folders created successfully")


# In[2]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install seaborn')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[4]:


(X_train,y_train),(X_test,y_test)=mnist.load_data()

print(X_train.shape)
print(X_test.shape)


# In[5]:


plt.figure(figsize=(10,5))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i],cmap='gray')
    plt.title(y_train[i])

plt.show()


# In[6]:


X_train=X_train/255.0
X_test=X_test/255.0


# In[7]:


X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)


# In[8]:


print(X_train.shape)


# In[21]:


model=Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(10,activation='softmax'))


# In[22]:


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# In[23]:


history=model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.2
)


# In[24]:


loss,accuracy=model.evaluate(
    X_test,
    y_test
)

print("Accuracy:",accuracy)


# In[25]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")

plt.legend(
    ['Train','Validation']
)

plt.show()


# In[26]:


predictions=model.predict(X_test)

predicted=np.argmax(
    predictions,
    axis=1
)


# In[27]:


plt.figure(figsize=(12,6))

for i in range(10):
    plt.subplot(2,5,i+1)

    plt.imshow(
        X_test[i].reshape(28,28),
        cmap='gray'
    )

    plt.title(
        f"P:{predicted[i]}"
    )

plt.show()


# In[28]:


print(
    classification_report(
        y_test,
        predicted
    )
)


# In[29]:


import seaborn as sns

cm=confusion_matrix(
    y_test,
    predicted
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")

plt.show()


# In[30]:


model.save(
    "mnist_cnn.h5"
)


# In[33]:


import os

os.makedirs("model", exist_ok=True)

model.save("model/mnist_cnn.h5")

print("Model saved successfully")


# In[34]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(['Train','Validation'])

plt.savefig("screenshots/accuracy.png")

plt.show()


# In[35]:


plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.savefig("screenshots/confusion_matrix.png")

plt.show()


# In[37]:


plt.savefig("screenshots/predictions.png")
plt.show()


# In[38]:


report = classification_report(
    y_test,
    predicted
)

print(report)


# In[39]:


with open("report/classification_report.txt", "w") as f:
    f.write(report)


# In[40]:


plt.figure(figsize=(12,6))

for i in range(10):
    plt.subplot(2,5,i+1)

    plt.imshow(
        X_test[i].reshape(28,28),
        cmap='gray'
    )

    plt.title(
        f"Actual:{y_test[i]}\nPred:{predicted[i]}"
    )

    plt.axis('off')

# Save image to screenshots folder
plt.savefig(
    "screenshots/predictions.png",
    bbox_inches='tight'
)

plt.show()


# In[41]:


import os

os.makedirs("screenshots", exist_ok=True)


# In[42]:


import os

os.makedirs("dataset", exist_ok=True)


# In[43]:


from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[44]:


import numpy as np

np.save("dataset/X_train.npy", X_train)
np.save("dataset/y_train.npy", y_train)

np.save("dataset/X_test.npy", X_test)
np.save("dataset/y_test.npy", y_test)

print("Dataset saved successfully!")


# In[45]:


import os

print(os.listdir("dataset"))


# In[ ]:




