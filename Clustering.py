import numpy as np
import os
import cv2
import keras
from jinja2.filters import K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Input
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.models import load_model


DATADIR= "S:\\Hi5\\Accesia\\ExportedXrays"
shapex=60
shapey=60
encoded_dim = 32
savepath='S:\\Hi5\\Accesia\\encoder1000.h5'
loadpath='S:\\Hi5\\Accesia\\encoder1000.h5'
k=25 #number of clusters
cluster0=[]
cluster1=[]
cluster2=[]
cluster3=[]
cluster4=[]
cluster5=[]
cluster6=[]
cluster7=[]
cluster8=[]
cluster9=[]
cluster10=[]
cluster11=[]
cluster12=[]
cluster13=[]
cluster14=[]
cluster15=[]
cluster16=[]
cluster17=[]
cluster18=[]
cluster19=[]



#Preparing image Data set by labels and keys
def prepdata(DATADIR):
    training_data = []
    Label_Data = []
    inputimages = []

    path = os.path.join(DATADIR)
    for img in os.listdir(path):
            # print(img)
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            file_name = os.path.basename(DATADIR)
            u = (os.path.splitext(img))
            new_array = cv2.resize(img_array, (shapex, shapey))
            new_array = ([new_array, u[0]])
            Label_Data.append(u[0])
            training_data.append(new_array)

    for train in training_data:
        training_data1 = np.array(train[0]).reshape(-1, shapex, shapey, 1)
        training_data1 = training_data1.astype('float32') / 255.0  # Normalize the pixel values
        # cv2.imshow("image",training_data1[1])
        inputimages.append(training_data1)
    inputimages = np.array(inputimages).reshape(-1, shapex, shapey, 1)

    return  inputimages, Label_Data


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

def elbowmethod(encoded_data,k):
    wcss = []
    noc=[]
    for i in range(1, k):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        cluster_assignments = kmeans.fit_predict(encoded_data)
        wcss.append(kmeans.inertia_)
        noc.append(i)

    plt.scatter(noc,wcss)
    plt.xlabel('number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow method')
    plt.show()


def load(loadpath):
    return load_model(loadpath)

def predict(encoder_model,inputimages):
    encoded_data = encoder_model.predict(inputimages)
    print(encoded_data.shape)
    encoded_data1=np.reshape(encoded_data,(len(inputimages),shapex*shapey))
    return encoded_data1

def clustering(k,encoded_data,Label_Data,cluster0,cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7,cluster8,cluster9,cluster10,cluster11,cluster12,cluster13,cluster14,cluster15,cluster16,cluster17,cluster18,cluster19):


    kmeans = KMeans(n_clusters=k)
    cluster_assignments = kmeans.fit_predict(encoded_data)

    # Step 4: Associate image names with cluster assignments

    name_cluster_mapping = {Label_Data[i]: cluster_assignments[i] for i in range(len(Label_Data))}
    #print("check this",name_cluster_mapping)

    # Step 5: Assign labels to clusters
    cluster_labels = [[] for _ in range(k)]  # List to store image names for each cluster

    for image_name, cluster in name_cluster_mapping.items():
        cluster_labels[cluster].append(image_name)
    print("zzzzz",cluster_labels)

    # Print the cluster labels
    for cluster_idx, labels in enumerate(cluster_labels):
        print(f"Cluster {cluster_idx}: {', '.join(labels)}")

        if cluster_idx == 0:
            for i in labels:
                cluster0.append(i)
        elif cluster_idx ==1:
            for i in labels:
                cluster1.append(i)
        elif cluster_idx == 2:
            for i in labels:
                cluster2.append(i)
        elif cluster_idx == 3:
            for i in labels:
                cluster3.append(i)
        elif cluster_idx == 4:
            for i in labels:
                cluster4.append(i)
        elif cluster_idx == 5:
            for i in labels:
                cluster5.append(i)
        elif cluster_idx == 6:
            for i in labels:
                cluster6.append(i)
        elif cluster_idx == 7:
            for i in labels:
                cluster7.append(i)
        elif cluster_idx == 8:
            for i in labels:
                cluster8.append(i)
        elif cluster_idx == 9:
            for i in labels:
                cluster9.append(i)
        elif cluster_idx == 10:
            for i in labels:
                cluster10.append(i)
        elif cluster_idx == 11:
            for i in labels:
                cluster11.append(i)
        elif cluster_idx == 12:
            for i in labels:
                cluster12.append(i)
        elif cluster_idx == 13:
            for i in labels:
                cluster13.append(i)
        elif cluster_idx == 14:
            for i in labels:
                cluster14.append(i)
        elif cluster_idx == 15:
            for i in labels:
                cluster15.append(i)
        elif cluster_idx == 16:
            for i in labels:
                cluster16.append(i)
        elif cluster_idx == 17:
            for i in labels:
                cluster17.append(i)
        elif cluster_idx == 18:
            for i in labels:
                cluster18.append(i)
        elif cluster_idx == 19:
            for i in labels:
                cluster19.append(i)
    return cluster0,cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7,cluster8,cluster9,cluster10,cluster11,cluster12,cluster13,cluster14,cluster15,cluster16,cluster17,cluster18,cluster19,cluster_assignments

def clusterfolders(cluster0,cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7,cluster8,cluster9,cluster10,cluster11,cluster12,cluster13,cluster14,cluster15,cluster16,cluster17,cluster18,cluster19,DATADIR):

    for l in cluster0:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster0" + "\\" + s, s_array)
    for l in cluster1:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster1" + "\\" + s, s_array)
    for l in cluster2:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster2" + "\\" + s, s_array)
    for l in cluster3:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster3" + "\\" + s, s_array)
    for l in cluster4:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster4" + "\\" + s, s_array)
    for l in cluster5:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster5" + "\\" + s, s_array)
    for l in cluster6:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster6" + "\\" + s, s_array)
    for l in cluster7:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster7" + "\\" + s, s_array)
    for l in cluster8:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster8" + "\\" + s, s_array)
    for l in cluster9:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster9" + "\\" + s, s_array)
    for l in cluster10:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster10" + "\\" + s, s_array)
    for l in cluster11:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster11" + "\\" + s, s_array)
    for l in cluster12:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster12" + "\\" + s, s_array)
    for l in cluster13:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster13" + "\\" + s, s_array)
    for l in cluster14:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster14" + "\\" + s, s_array)
    for l in cluster15:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster15" + "\\" + s, s_array)
    for l in cluster16:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster16" + "\\" + s, s_array)
    for l in cluster17:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster17" + "\\" + s, s_array)
    for l in cluster18:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster18" + "\\" + s, s_array)
    for l in cluster19:
        s = l + '.jpg'
        # print(DATADIR1+'\\'+s)
        s_array = cv2.imread(DATADIR + '\\' + s)
        cv2.imwrite("S:\\Hi5\\Accesia\\Clusteredimages1000\\Cluster19" + "\\" + s, s_array)


def plot(encoded_data,cluster_assignments):
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=cluster_assignments)
    plt.xlabel('Encoded Dimension 1')
    plt.ylabel('Encoded Dimension 2')
    plt.title('Encoded Representations')
    plt.show()



inputimages,labeldata= prepdata(DATADIR)
# encoder_model,encoder_input,encoder_output=encoder()
# decoder_model=decoder(encoder_output)
# s=autoencoder(encoder_input,encoder_model,decoder_model,inputimages)
#
#
#
#
#
# for i in range(0,4):
#     plt.figure(1)
#     plt.imshow(s[i],cmap='gray')
#     plt.figure(2)
#     plt.imshow(inputimages[i],cmap='gray')
#
#     plt.show()
#
# save(encoder_model,savepath)

#
encoded_Data=predict(load(loadpath),inputimages)
elbowmethod(encoded_Data,k)
c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,ca=clustering(k,encoded_Data,labeldata,cluster0,cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7,cluster8,cluster9,cluster10,cluster11,cluster12,cluster13,cluster14,cluster15,cluster16,cluster17,cluster18,cluster19)
# clusterfolders(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,DATADIR)
plot(encoded_Data,ca)
