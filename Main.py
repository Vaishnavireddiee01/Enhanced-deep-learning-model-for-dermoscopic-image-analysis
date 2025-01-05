import tensorflow as tf
import efficientnet.tfkeras as efn
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import efficientnet.keras as efn
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pickle
from keras.models import model_from_json, Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.layers import Input, Dense, Flatten
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.transform import resize 
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle


from PIL import Image, ImageTk 
from keras.preprocessing.image import img_to_array
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import cv2
from skimage import color
from skimage.feature import greycomatrix, greycoprops
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


global filename
global X,Y
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test
global cnn
global labels

labels = ['actinic keratosis','basal cell carcinoma',
          'dermatofibroma','melanoma','nevus',
          'pigmented benign keratosis','seborrheic keratosis',
          'squamous cell carcinoma','vascular lesion']


with open('model/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    cnn_classifier = model_from_json(loaded_model_json)
json_file.close()    
cnn_classifier.load_weights("model/model_weights.h5")
cnn_classifier._make_predict_function() 


main = tkinter.Tk()
main.title("Skin Cancer Detection using Deep EfficientNet50") 
main.geometry("1300x1200")

def remove_green_pixels(image):
  # Transform from (256,256,3) to (3,256,256)
  channels_first = channels_first_transform(image)

  r_channel = channels_first[0]
  g_channel = channels_first[1]
  b_channel = channels_first[2]

  # Set those pixels where green value is larger than both blue and red to 0
  mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
  channels_first = np.multiply(channels_first, mask)

  # Transfrom from (3,256,256) back to (256,256,3)
  image = channels_first.transpose(1, 2, 0)
  return image

def rgb2lab(image):
  return color.rgb2lab(image)

def rgb2gray(image):
  return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0], squeeze=False): #extract glcm features
  single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
  gclm = greycomatrix(single_channel_image, offsets, angles)
  return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image): 
  image = channels_first_transform(image).reshape(3,-1)

  r_channel = image[0]
  g_channel = image[1]
  b_channel = image[2]

  r_hist = np.histogram(r_channel, bins = 26, range=(0,255))[0]
  g_hist = np.histogram(g_channel, bins = 26, range=(0,255))[0]
  b_hist = np.histogram(b_channel, bins = 26, range=(0,255))[0]

  return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
  color_histogram = np.histogram(image.flatten(), bins = 255, range=(0,255))[0]
  return np.array([
    np.mean(color_histogram),
    np.std(color_histogram),
    stats.entropy(color_histogram),
    stats.kurtosis(color_histogram),
    stats.skew(color_histogram),
    np.sqrt(np.mean(np.square(color_histogram)))
  ])

def texture_features(full_image, offsets=[1], angles=[0], remove_green = True):
  image = remove_green_pixels(full_image) if remove_green else full_image
  gray_image = rgb2gray(image)
  glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
  return glcm_features(glcmatrix)

def glcm_features(glcm):
  return np.array([
    greycoprops(glcm, 'correlation'),
    greycoprops(glcm, 'contrast'),
    greycoprops(glcm, 'energy'),
    greycoprops(glcm, 'homogeneity'),
    greycoprops(glcm, 'dissimilarity'),
  ]).flatten()

def channels_first_transform(image):
  return image.transpose((2,0,1))

def extract_features(image):
  offsets=[1,3,10,20]
  angles=[0, np.pi/4, np.pi/2]
  channels_first = channels_first_transform(image)
  return np.concatenate((
      texture_features(image, offsets=offsets, angles=angles),
      texture_features(image, offsets=offsets, angles=angles, remove_green=False),
      histogram_features_bucket_count(image),
      histogram_features(channels_first[0]),
      histogram_features(channels_first[1]),
      histogram_features(channels_first[2]),
      ))

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index 

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')    
    text.insert(END,"Different types of skin cancer disease Found in Dataset : "+str(labels)+"\n\n") 
    text.insert(END,"Total types  are : "+str(len(labels)))


def featuresExtraction():
    global filename
    global X,Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.npy"):
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (64,64))
                    class_label = getID(name)
                    features = extract_features(img)
                    Y.append(class_label)
                    X.append(features)
                    print(name+" "+root+"/"+directory[j]+" "+str(features.shape)+" "+str(class_label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X",X)
        np.save("model/Y",Y)
    X = X.astype('float32')
    X = X/255 #features normalization
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Extracted GLCM & Texture Features : "+str(X[0])+"\n\n")
    #text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset train & test split. 80% dataset images used for training and 20% for testing\n\n")
    text.insert(END,"80% training images : "+str(X_train.shape[0])+"\n\n")
    text.insert(END,"20% testing images : "+str(X_test.shape[0])+"\n\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(10, 8)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()



def runEfficientNet():
    global X_train, X_test, y_train, y_test, X, Y, efficientNet
    global accuracy, precision, recall, fscore

    # Check the shape of X
    print(f"Shape of X before reshaping: {X.shape}")

    # Ensure X has the expected dimensions
    if len(X.shape) == 2:
        XX = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Add channel dimension
    elif len(X.shape) == 3:
        XX = X  # X already has the correct shape
    else:
        raise ValueError(f"Unexpected shape for X. Expected 2 or 3 dimensions, got {len(X.shape)}.")

    print(f"Shape of X after reshaping: {XX.shape}")

    Y1 = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(XX, Y1, test_size=0.2)
    
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            efficientNet = model_from_json(loaded_model_json, custom_objects={'EfficientNetB0': efn.EfficientNetB0})
        json_file.close()
        efficientNet.load_weights("model/model_weights.h5") 
    else:
        base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(XX.shape[1], XX.shape[2], XX.shape[3]))
        efficientNet = Sequential()
        efficientNet.add(base_model)
        efficientNet.add(Flatten())
        efficientNet.add(Dense(units=256, activation='relu'))  
        efficientNet.add(Dense(units=Y1.shape[1], activation='softmax'))  
        efficientNet.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        
        hist = efficientNet.fit(X_train, y_train, batch_size=12, epochs=10, shuffle=True, verbose=2)
        
        efficientNet.save_weights('model/model_weights.h5')  
        model_json = efficientNet.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        
        with open('model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
    
    print(efficientNet.summary())
    
    # Reshape X_test to match model input
    X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension if missing
    print(f"Shape of X_test after reshaping: {X_test.shape}")
    
    predict = efficientNet.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    calculateMetrics("EfficientNetB0", predict, y_test)



def graph():
    df = pd.DataFrame([['EfficientNet-50','Accuracy',accuracy[0]],
                       ['EfficientNet-50','Precision',precision[0]],
                       ['EfficientNet-50','Recall',recall[0]],
                       ['EfficientNet-50','FScore',fscore[0]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()



def predict():
    global efficientNet
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    test = []
    img = cv2.resize(img, (64,64))
    features = extract_features(img)
    test.append(features)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict = efficientNet.predict(test)
    predict = np.argmax(predict)

    disease_label = labels[predict]

    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, f'Skin cancer type is: {disease_label}', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow(f'Skin cancer type is: {disease_label}', img)
    cv2.waitKey(0)




bg_img = Image.open("image.jpg")  
bg_img = bg_img.resize((1300, 1200), Image.ANTIALIAS)  

bg_img_tk = ImageTk.PhotoImage(bg_img)

bg_label = Label(main, image=bg_img_tk)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)


font = ('times', 16, 'bold')
title = Label(main, text='Skin Cancer Detection using Deep EfficientNet50')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=150)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Kaggle Dataset", command=uploadDataset)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)  

featuresButton = Button(main, text="Extract Features", command=featuresExtraction)
featuresButton.place(x=50,y=250)
featuresButton.config(font=font1) 


cnnButton = Button(main, text="Run EfficientNet  Algorithm", command=runEfficientNet)
cnnButton.place(x=50,y=350)
cnnButton.config(font=font1)

graphButton = Button(main, text="Performance Graph", command=graph)
graphButton.place(x=50,y=450)
graphButton.config(font=font1)


predictButton = Button(main, text="Skin Cancer type Detection from Test Image", command=predict)
predictButton.place(x=50,y=550)
predictButton.config(font=font1)


#main.config(bg='LightSkyBlue')
main.mainloop()
