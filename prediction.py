import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

from PIL import Image
import requests
from io import BytesIO

import time
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity

import random

join_table = pd.read_csv('mac_join_table.csv')
y_table = pd.read_csv('all_one_hot_encoded.csv')

def read_img(url, size = (224,224)):
    """Read and resize image from url
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = image.img_to_array(img)
    return img

def show_image(index):
    """ Shows image from specified index"""
    fig, ax = plt.subplots(1, figsize=(4, 4))
    img = image.load_img(join_table.loc[index].values[0], target_size=(224, 224))
    ax.imshow(img)
    ax.text(0,200, 'LABEL: %s' % join_table.loc[index].values[2], backgroundcolor = 'w')
    ax.axis('off')
    return plt.show()

def show_image_pred(url, pred):
    """ Shows image from specified url"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    ax.text(0.1,0.1, 'PRED LABEL: %s' % list(y_table.columns)[int(pred)], backgroundcolor = 'w')
    ax.axis('off')
    return plt.show()


model_RN50 = load_model('model_RN50.h5')
model_2lr = load_model('model_2lr.h5')
model_2cr = load_model('model_2cr.h5')
model_2dr = load_model('model_2dr.h5')
model_2gr = load_model('model_2gr.h5')
model_2jr = load_model('model_2jr.h5')


dog_features = pd.read_csv('comined_features_updated.csv')
dog_features.head()
dog_features.columns = ['label_name', 'Energy level', 'Exercise requirements', 'Playfulness',
       'Affection level', 'Friend to dogs', 'Friend to other pets', 'Friend to strangers',
       'Watchfulness', 'Easy training', 'Grooming', 'Heat sensitive',
       'Vocality', 'Jaro_Winkler']

def find_features(label):
    """ Returns dog characteristics """
    feature_names = dog_features.columns[1:-1]
    find_features = dog_features.loc[dog_features['label_name'] == label].values[0][1:-1]
    print('\n',label)
    for i, feature in enumerate(feature_names):
        print(f'\n {feature}: {find_features[i]}/5')
    return find_features

def find_features_table(labels):
    """ Returns dogs characteristics table"""
    df = dog_features.copy()
    display(df.loc[df['label_name'].isin(labels)].set_index('label_name').drop(labels='Jaro_Winkler', axis=1))

train_breeds_names = pd.read_csv('train_breeds_names.csv', header = None)
train_breeds_features = np.load('train_arr_rn50.npy')

def show_rand_image(label):
    """ Returns random picture with specific label """
    if type(label) == str:
        label_indexes = join_table[join_table['labels'] == label].index
        rand_image_idx = random.randint(label_indexes.min(), label_indexes.max())
        fig, ax = plt.subplots(1, figsize=(4, 4))
        img = image.load_img(join_table.loc[rand_image_idx].values[0], target_size=(224, 224))
        ax.text(0.9,0.1, '%s' % label, backgroundcolor = 'w')
        ax.imshow(img)
        ax.axis('off')
    else:
        cnt = len(label)
        fig = plt.figure(1,figsize=(16,16))
        grid = ImageGrid(fig,111,nrows_ncols=(1,cnt), axes_pad = 0.05)
        for i, lab in enumerate(label):
            label_indexes = join_table[join_table['labels'] == lab].index
            rand_image_idx = random.randint(label_indexes.min(), label_indexes.max())
            ax = grid[i]
            img = image.load_img(join_table.loc[rand_image_idx].values[0], target_size=(224, 224))
            ax.text(0.9,0.1, '%s' % lab, backgroundcolor = 'w')
            ax.imshow(img)
            ax.axis('off')
    return plt.show()

def predict_dog_label():
    """Predicts the dog breed and recommends similar dogs based on its characterictics"""
    url = input ("Enter img url :")
    image_to_tensor = np.zeros(((1), 224, 224, 3), dtype='float32')
    
    # converts image to centralized format for ResNet model
    image_to_tensor[0] = preprocess_input(np.expand_dims(read_img(url).copy(), axis=0))
    
    # predicts bottleneck features using ResNet model
    get_image_features = model_RN50.predict(image_to_tensor, verbose=1)
    
    # predicts image label based on features from ResNet model
    pred = ((model_2cr.predict(get_image_features)+model_2dr.predict(get_image_features)+
             model_2gr.predict(get_image_features)+model_2jr.predict(get_image_features)+
             model_2lr.predict(get_image_features))/5)
    
    # returns top 5 predicted labels
    top_5 = [i[0] for i in sorted(list(zip(list(y_table.columns),pred.tolist()[0])),
             key = lambda x: x[1], reverse = True)[:6]]
    label = list(y_table.columns)[int(pred.argmax(axis=1))]
    show_image_pred(url, pred.argmax(axis=1))
    
    # finds most similar picture in local database based on cosine similarity of image features
    similar_pic = sorted(list(zip(list(train_breeds_names[0]),
                                  (cosine_similarity(get_image_features,pd.DataFrame(train_breeds_features))).reshape(-1, 1))), 
                    key=lambda x: x[1], reverse = True)[1][0]
    time.sleep(0.5)
    sim = input('\nDo you want the most similar pic in database: (y/n) ')
    if sim == 'y':
        print('\n Most similar pic in database: \n')
        show_image(similar_pic)
    time.sleep(0.5)
    print('\nTop 5 visually most similar dogs:', top_5[1:])
    show = input('\nDo you want to see those dogs?: (y/n) ')
    if show =='y':
        show_rand_image(top_5[1:])
    feat = input("\nDo you want to see the features of your dog? (y,n) ")
    if feat == 'y':
        if dog_features.loc[dog_features['label_name'] == label].values[0][1:-1][0] == 0:
            print('\nThis dog is not a pet or no information available\n')
            feat_vis = input('\nDo you want to see the features of visually similar dogs? (y/n) ')
            if feat_vis == 'y':
                find_features_table(top_5[1:])
                return label
        find_features_table([label])
    feat_sim = input('\nDo you want to find similar dogs based on its features? (y/n) ')
    if feat_sim == 'y':
        print('\n')
        # finds most similar dog based on cosine similarity of dog characteristcs
        top_5_feat = [x[0] for x in sorted(list(zip(list(dog_features.label_name),
                      (cosine_similarity(dog_features[dog_features.columns[1:-1]])[int(pred.argmax(axis=1))]))), 
                        key=lambda x: x[1], reverse = True)[1:6]]
        find_features_table(top_5_feat)
        feat_sim2 = input("\nDo you want to see them? (y,n) ")
        if feat_sim2 == 'y':
            show_rand_image(top_5_feat)
    return label
