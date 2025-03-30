"""
Face Decoder Stuff

"""

import numpy as np
import time
import matplotlib.pyplot as plt
import cv2 
import os
#import tensorflow as tf
import random
from sklearn.decomposition import PCA
import matplotlib.image as img
import png
from PIL import Image

# Displays an image
def displayImage(image):
    plt.imshow(image, interpolation='nearest') #display the image 
    plt.gray()  #grayscale conversion
    plt.show()



WIDTH = 44
HEIGHT = 50
# Load Image datasets as flattened numpy arrays
# 150 by 130  ->  19500
# 100 by 87 -> 8700
# 50 by 44 -> 2200
def loadData(path):
    data = []
    valid_images = [".jpg",".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        print(os.path.join(path,f));
        frame = np.asarray(img.imread(os.path.join(path,f)))        
        image = cv2.resize(frame,(WIDTH,HEIGHT))
        
        #threshold = int(np.mean(image))
        (thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        data.append(np.ndarray.flatten(image))
    return np.asarray(data)


def ditherImage(filename):
    img = Image.open(filename).convert('RGB').resize((WIDTH,HEIGHT))
    #img = cv2.resize(im,(WIDTH,HEIGHT))
    #newimg = img.convert(mode='P',dither=Image.Dither.FLOYDSTEINBERG, palette=Image.Palette.ADAPTIVE, colors=2)
    newimg = img.convert(mode='P',dither=Image.FLOYDSTEINBERG, palette=Image.ADAPTIVE, colors=2)
    (thresh, image) = cv2.threshold(np.array(newimg)*255, 127, 255, cv2.THRESH_BINARY_INV)
    return image

def loadData2(path):
    data = []
    valid_images = [".jpg",".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
              
        image = ditherImage(os.path.join(path,f))
        
        data.append(np.ndarray.flatten(image))
    return np.asarray(data)



#%%
#Load Flattened Training and Testing Data
print("Loading Data")
#train_images = loadData("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/Train1")
train_images = loadData2("C:/Users/jesse/Documents/C64 Book/ByteCamp_11_2024/DeepFake/Train1")

#%%
print("Mean Centering Data")
meanFace = np.mean(train_images,axis=0) # Mean face
train_data = (train_images - meanFace)  # Mean centered



#%%
print("Reduce dimensionality and normalize")
PARAM_SIZE = 5
OUTPUT_SIZE = WIDTH*HEIGHT
pca = PCA(n_components=PARAM_SIZE)
train_dataK = pca.fit_transform(train_data)
train_dataK = (train_dataK/np.linalg.norm(train_dataK,axis=0))

# Get components
m = meanFace
f0 = pca.components_[0]
f1 = pca.components_[1]
f2 = pca.components_[2]
f3 = pca.components_[3]
f4 = pca.components_[4]



# Print faces
steps = np.arange(-1.0,1.0,0.01)
normMean = (meanFace/np.linalg.norm(meanFace,axis=0))

for i in range(5):
    count = 0
    for x in steps:
        feats = np.asarray([0.0,0.0,0.0,0.0,0.0])
        feats[i]=x

        test = f0*feats[0] + f1*feats[1] + f2*feats[2] + f3*feats[3] + f4*feats[4] + normMean
        test = test*255
        image = (test.reshape(HEIGHT,WIDTH))
        # Threshold is at the average darkness
        #(thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        threshold = int(np.mean(image))
        
        (thresh, image) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        cv2.imwrite("C:/Users/jesse/Documents/C64 Book/New Features/Feature "+str(i+1)+"/"+str(count)+".png",cv2.resize(image,(WIDTH*10,HEIGHT*10), interpolation=cv2.INTER_NEAREST))
        count += 1


#%%


#face = f0*w0 + f1*w1 + f2*w2 + f3*w3 + f4*w4 + mean_scaled

f0_scaled = ((f0*100000.0)/32.0).astype(int)
f1_scaled = ((f1*100000.0)/32.0).astype(int)
f2_scaled = ((f2*100000.0)/32.0).astype(int)
f3_scaled = ((f3*100000.0)/32.0).astype(int)
f4_scaled = ((f4*100000.0)/32.0).astype(int)
mean_scaled = (normMean*100000.0).astype(int)

# So this gives 64 options for each vector.
w0 = 32
w1 = 0
w2 = 0
w3 = 0
w4 = 0

face = f0_scaled*w0 + f1_scaled*w1 + f2_scaled*w2 + f3_scaled*w3 + f4_scaled*w4 + mean_scaled
face = (face.reshape(HEIGHT,WIDTH)).astype(float)
threshold = int(np.mean(face))
(thresh, image) = cv2.threshold(face, threshold, 255, cv2.THRESH_BINARY)
plt.imshow(image, cmap='gray')

np.savetxt("C:/Users/jesse/Documents/C64 Book/weights0.txt", f0_scaled, delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/jesse/Documents/C64 Book/weights1.txt", f1_scaled, delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/jesse/Documents/C64 Book/weights2.txt", f2_scaled, delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/jesse/Documents/C64 Book/weights3.txt", f3_scaled, delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/jesse/Documents/C64 Book/weights4.txt", f4_scaled, delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/jesse/Documents/C64 Book/mean.txt", mean_scaled, delimiter =", ",fmt="%i,",newline=" ")

#%%
# Buid Model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, SpatialDropout2D
from keras.layers.embeddings import Embedding

print("Building Model")
model = Sequential()
model.add(Dense(OUTPUT_SIZE, input_dim=PARAM_SIZE, activation='sigmoid')) # Note the input = 20 because we have 20 conditions/symptoms we're using

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x=train_dataK,y=train_data, epochs=20)


# Save Model
# model.save('FaceGen_44x50')
# Load Model
# modelx = keras.models.load_model('FaceGen_44x50')
#%%

pred = model.predict(train_dataK)

# Display 10 approximations
for i in range(10):
    image = (pred[i].reshape(HEIGHT,WIDTH) + normMean.reshape(HEIGHT,WIDTH))*255
    # Threshold is at the average darkness
    threshold = int(np.mean(image))
    (thresh, image) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    displayImage(image)
    
#%%

# Save features

steps = np.arange(-0.05,0.05,0.001)

for i in range(10):
    count = 0
    for x in steps:
        feats = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
        feats[0][i]=x
        test = model.predict(feats)
        image = (test.reshape(HEIGHT,WIDTH) + normMean.reshape(HEIGHT,WIDTH))*255
        # Threshold is at the average darkness
        threshold = int(np.mean(image))
        (thresh, image) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/Features/Feature "+str(i+1)+"/"+str(count)+".png", image)
        count += 1

#%%






#%%
      
# Do it with just martix math
        
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def relu(x):
    if(x < 0):
        return 0
    else:
        return x

# Test
w = model.get_weights()[0]
b = model.get_weights()[1]
f = np.asarray([-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

prod = np.dot(f,w)+b
sig_vectorized = np.vectorize(sigmoid)
#print(sig_vectorized(prod))
image = sig_vectorized(prod).reshape(HEIGHT,WIDTH)*255 + normMean.reshape(HEIGHT,WIDTH)*255
threshold = int(np.mean(image))
(thresh, image) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
displayImage(image)

#%%
      
# Do it with just martix math
# Only Integers though
# 25 by 22        
# 4-9-2022

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def sigmoid_int(x):
    return (1/(1 + (27^(-x*10))/10))

def relu(x):
    if(x < 0):
        return 0
    else:
        return x
    
def linear(x):
    return x

# Test
w = (model.get_weights()[0]*52).astype(int)
b = (model.get_weights()[1]*52).astype(int)
f = (np.asarray([0.02,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])*10000).astype(int)

prod = (np.dot(f,w)+b)
sig_vectorized = np.vectorize(relu)
#print(sig_vectorized(prod))
image = ((sig_vectorized((prod/10000).astype(int)).reshape(HEIGHT,WIDTH)) + (normMean.reshape(HEIGHT,WIDTH)*52).astype(int))
threshold = int(np.mean(image))
#(thresh, image_t) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
image[image>threshold] = 255
image[image<255] = 0
displayImage(image)

#%%
      
# Do it with just martix math
# Only Integers though
# 50 by 44        
# 4-9-2022

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def sigmoid_int(x):
    return (1/(1 + (27^(-x*10))/10))

def relu(x):
    if(x < 0):
        return 0
    else:
        return x
    
def linear(x):
    return x

# Test
w = (model.get_weights()[0]*52).astype(int)
b = (model.get_weights()[1]*52).astype(int)
f = (np.asarray([0.02,0.0,0.0,0.0,0.0])*10000).astype(int)

prod = (np.dot(f,w)+b)
sig_vectorized = np.vectorize(relu)
#print(sig_vectorized(prod))
image = ((sig_vectorized((prod/10000).astype(int)).reshape(HEIGHT,WIDTH)) + (normMean.reshape(HEIGHT,WIDTH)*52).astype(int))
threshold = int(np.mean(image))
#(thresh, image_t) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
image[image>threshold] = 255
image[image<255] = 0
displayImage(image)
#%%
# 	-128 to 127

w_scaled = w
b_scaled = b
mean_scaled = (normMean*52).astype(int)




np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights0.txt", w_scaled[0], delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights1.txt", w_scaled[1], delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights2.txt", w_scaled[2], delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights3.txt", w_scaled[3], delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights4.txt", w_scaled[4], delimiter =", ",fmt="%i,",newline=" ")
'''
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights5.txt", w_scaled[5], delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights6.txt", w_scaled[6], delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights7.txt", w_scaled[7], delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights8.txt", w_scaled[8], delimiter =", ",fmt="%i,",newline=" ")
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/weights9.txt", w_scaled[9], delimiter =", ",fmt="%i,",newline=" ")
'''
np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/bias.txt", b_scaled, delimiter =", ",fmt="%i,",newline=" ")

np.savetxt("C:/Users/Jesse/Documents/NIST/Personal Projects/DeepFake/meanFace.txt", mean_scaled, delimiter =", ",fmt="%i,",newline=" ")