import numpy as np
import os
import cv2
import keras
#from jinja2.filters import K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Input
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, ShuffleSplit
shapex=60
shapey=60
savepath='S:\\Hi5\\Accesia\\edgeautoclass.h5'
loadpath='S:\\Hi5\\Accesia\\autoclass.h5'
test_image_folder_path = r"S:\Hi5\Accesia\Test"




# Set the path to the folder containing the image folders


# Function to load and preprocess the images
def load_images_from_folder(folder):
    images = []
    labels = []
    inputimages=[]
    for class_folder in os.listdir(folder):
       # print(class_folder)
        class_folder_path = os.path.join(folder, class_folder)
        for filename in os.listdir(class_folder_path):
            img_path = os.path.join(class_folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (shapex, shapey))
                images.append(img)
                labels.append(class_folder)
    for train in images:
        training_data1 = np.array(train).reshape(-1, shapex, shapey, 1)
        training_data1 = training_data1.astype('float32') / 255.0  # Normalize the pixel values
        # cv2.imshow("image",training_data1[1])
        inputimages.append(training_data1)
    inputimages = np.array(inputimages).reshape(-1, shapex, shapey, 1)

   #print("Total Images:", len(inputimages))
    #print("Total Labels:", len(labels))
    return inputimages, labels

# Load the images and labels




def load_testimages_from_folder(folder):
    gtlabels = []
    images = []
    inputimages = []
    for class_folder in os.listdir(folder):
        # print(class_folder)
        class_folder_path = os.path.join(folder, class_folder)
        for filename in os.listdir(class_folder_path):
            img_path = os.path.join(class_folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (shapex, shapey))
                images.append(img)
                gtlabels.append(class_folder)
    for train in images:
        training_data1 = np.array(train).reshape(-1, shapex, shapey, 1)
        training_data1 = training_data1.astype('float32') / 255.0  # Normalize the pixel values
        # cv2.imshow("image",training_data1[1])
        inputimages.append(training_data1)
    testimages = np.array(inputimages).reshape(-1, shapex, shapey, 1)
    return testimages, gtlabels







def encoder():
    encoder_input = keras.Input(shape=(shapex, shapey, 1))
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(encoder_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    #x = Flatten()(x)
    encoded_output = x

    encoder_model = Model(inputs=encoder_input, outputs=encoded_output)
    return encoder_model,encoder_input,encoded_output

def decoder(encoded_ouput):
    decoder_input = encoded_ouput

    # x = Dense(shapex * shapey * 1, activation='relu')(decoder_input)
    #
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    decoded_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder_model = Model(inputs=decoder_input, outputs=decoded_output)
    return decoder_model

def autoencoder(encoder_input,encoder_model,decoder_model,inputimages):
    autoencoder_input = encoder_input
    autoencoder_output = decoder_model(encoder_model(autoencoder_input))
    autoencoder_model = Model(inputs=autoencoder_input, outputs=autoencoder_output)

    autoencoder_model.compile(optimizer='adam', loss='mse')
    autoencoder_model.summary()

    autoencoder_model.fit(inputimages, inputimages, batch_size=1, epochs=50, verbose=1)
    return autoencoder_model.predict(inputimages)

def save(encoder_model,savepath):
    encoder_model.save(savepath)
def load(loadpath):
    return load_model(loadpath)

def predict(encoder_model,inputimages):
    encoded_data = encoder_model.predict(inputimages)
    #print(encoded_data.shape)
    encoded_data1=np.reshape(encoded_data,(len(inputimages),shapex*shapey))
    return encoded_data1

inputimages, labels = load_images_from_folder(image_folder_path)
testimages,gtlabels=load_testimages_from_folder(test_image_folder_path)
# labels=np.array(labels)
# gtlabels=np.array(gtlabels)
# inputimages=np.array(inputimages)
#inputimages,labeldata= prepdata(DATADIR)
# encoder_model,encoder_input,encoder_output=encoder()
# decoder_model=decoder(encoder_output)
# s=autoencoder(encoder_input,encoder_model,decoder_model,inputimages)
#
#
# for i in range(0,4):
#     plt.figure(1)
#     plt.imshow(s[i],cmap='gray')
#     plt.figure(2)
#     plt.imshow(inputimages[i],cmap='gray')
#     plt.show()

#save(encoder_model,savepath)

############################################################################# LOAD #########################################


encoded_Data=predict(load(loadpath),inputimages)
test_encoded_Data=predict(load(loadpath),testimages)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


label_encoder = LabelEncoder()
gtlabels = label_encoder.fit_transform(gtlabels)
print(gtlabels[2])

X_train=encoded_Data
y_train=labels
X_test=test_encoded_Data
y_test=gtlabels

print("before",X_test.shape)
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = y_train.ravel()
X_test = X_test.reshape(X_test.shape[0], -1)
y_test = y_test.ravel()

print(X_test.shape)
##########################Logistic Regression#######################################
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_class_pred = classifier.predict(X_test)

#print(y_class_pred)

classification_accuracy = accuracy_score(y_test, y_class_pred)
print("Logistic Regression Classification Accuracy:", classification_accuracy*100)


y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_class_pred)

misclassified_images = []
misclassified_labels = []
misclassified_predictions = []
misclassified_count = 0
for i in range(len(y_test_labels)):
    if y_test_labels[i] != y_pred_labels[i]:
        misclassified_count += 1
        misclassified_images.append(testimages[i])
        misclassified_labels.append(y_test_labels[i])
        misclassified_predictions.append(y_pred_labels[i])

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_class_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

for i in range(len(misclassified_images)):
    plt.imshow(misclassified_images[i], cmap='gray')
    plt.title(f'True Label: {misclassified_labels[i]}, Predicted Label: {misclassified_predictions[i]}')
    plt.show()
print("Total Misclassified Images:", misclassified_count)

#################################Random forest#######################

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

classification_accuracy = accuracy_score(y_test, y_pred)
print("Random forest Classification Accuracy:", classification_accuracy*100)


y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_class_pred)

misclassified_images = []
misclassified_labels = []
misclassified_predictions = []
misclassified_count = 0
for i in range(len(y_test_labels)):
    if y_test_labels[i] != y_pred_labels[i]:
        misclassified_count += 1
        misclassified_images.append(testimages[i])
        misclassified_labels.append(y_test_labels[i])
        misclassified_predictions.append(y_pred_labels[i])

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_class_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# for i in range(len(misclassified_images)):
#     plt.imshow(misclassified_images[i], cmap='gray')
#     plt.title(f'True Label: {misclassified_labels[i]}, Predicted Label: {misclassified_predictions[i]}')
#     plt.show()
print("Total Misclassified Images:", misclassified_count)



#############################SVC##########################################

classifier = SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classification_accuracy = accuracy_score(y_test, y_pred)
print("SVC Classification Accuracy:", classification_accuracy*100)


#Evaluate the classification performance
# classification_report = classification_report(y_test, y_pred)
# print("Classification Report:\n", classification_report)
##################################################################################
