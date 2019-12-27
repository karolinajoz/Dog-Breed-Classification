import streamlit as st
import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

from PIL import Image
import requests
from io import BytesIO

import glob
from sklearn.metrics.pairwise import cosine_similarity
import random


@st.cache
def read_img(url, size = (224,224)):
    """Read and resize image from url
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = image.img_to_array(img)
    return img

def show_image_pred(url, pred):
    """ Shows image from specified url"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    st.image(img, caption=list(y_table.columns)[int(pred)].replace("_"," ").title(),
         use_column_width=False)

def show_image(index):
    """ Shows image from specified index"""
    img = image.load_img(join_table.loc[index].values[0])
    st.image(img, caption=join_table.loc[index].values[2].replace("_"," ").title(),
         use_column_width=False)
    
def read_test_img(photo):
    """ Read and resize image from specified index from file"""
    img = image.load_img(f'./test_photos/{photo}')
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = image.img_to_array(img)
    return img

def show_test_img(photo):
    """ Shows image from file """
    img = image.load_img(f'./test_photos/{photo}')
    st.image(img, use_column_width=True)

def show_slide(index):
    """ Returns slide for slideshow from specified index"""
    img = image.load_img(slides[index])
    st.image(img, use_column_width=True)

@st.cache
def load_slides():
    """ Loads all slides paths """
    slides = (sorted(glob.glob('./Slides/*')))
    return slides

@st.cache
def read_files_and_models ():
    """ Returns required tables and models"""
    join_table = pd.read_csv('mac_join_table.csv')
    y_table = pd.read_csv('all_one_hot_encoded.csv') 
    model_2lr = load_model('model_2lr.h5')
    model_2cr = load_model('model_2cr.h5')
    model_2dr = load_model('model_2dr.h5')
    model_2gr = load_model('model_2gr.h5')
    model_2jr = load_model('model_2jr.h5')
    model_RN50 = load_model('model_RN50.h5')
    dog_features = pd.read_csv('comined_features_updated.csv')
    dog_features.columns = ['label_name', 'Energy level', 'Exercise requirements', 'Playfulness',
       'Affection level', 'Friend to dogs', 'Friend to other pets', 'Friend to strangers',
       'Watchfulness', 'Easy training', 'Grooming', 'Heat sensitive',
       'Vocality', 'Jaro_Winkler']
    train_breeds_names = pd.read_csv('train_breeds_names.csv', header = None)
    train_breeds_features = np.load('train_arr_rn50.npy')
    return join_table, y_table, model_RN50, model_2lr, model_2cr, model_2dr, model_2jr, model_2gr, dog_features,train_breeds_names,train_breeds_features
    
@st.cache
def find_features(label):
    """ Returns dog characteristics """
    feature_names = dog_features.columns[1:-1]
    find_features = dog_features.loc[dog_features['label_name'] == label].values[0][1:-1]
    print('\n',label)
    for i, feature in enumerate(feature_names):
        print(f'\n {feature}: {find_features[i]}/5')
    return find_features

@st.cache
def find_features_table(labels):
    """ Returns dogs characteristics table"""
    df = dog_features.copy()
    return (df.loc[df['label_name'].isin(labels)].set_index('label_name').drop(labels='Jaro_Winkler', axis=1)).transpose()
    
def show_rand_image(label):
    """ Returns random picture with specific label """
    if type(label) == str:
        label_indexes = join_table[join_table['labels'] == label].index
        rand_image_idx = random.randint(label_indexes.min(), label_indexes.max())
        img = image.load_img(join_table.loc[rand_image_idx].values[0])

        st.image(img, caption=label.replace("_"," ").title(), use_column_width=False)
    else:
        img_lst = []
        for i, lab in enumerate(label):
            label_indexes = join_table[join_table['labels'] == lab].index
            rand_image_idx = random.randint(label_indexes.min(), label_indexes.max())

            img = image.load_img(join_table.loc[rand_image_idx].values[0],target_size=(224, 224))
            img_lst.append(img)
        st.image(img_lst, caption=[lab.replace("_"," ").title() for lab in label], use_column_width=False)

#####################

st.title("Dog breed classification and recommendation app")

slides = load_slides()

what =  st.radio("What do you want to do?", ('Slideshow', 'App'))

if what == 'Slideshow':
    st.text('Slideshow')
    slideshow = st.slider('slide', 0, 13, 0) #min, max, default
    show_slide(slideshow)
    
else:
    img_path = st.radio("Where is your photo?", ("Local", "Online"))
    if img_path == "Local":
        name = st.text_input("Enter file name", 'charles-K4mSJ7kc0As-unsplash.jpg')
        img = read_test_img(name)
    else:
        url = st.text_input('Enter URL',"https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/adorable-cavalier-king-charles-spaniel-puppy-royalty-free-image-523255012-1565106446.jpg?crop=0.448xw:1.00xh;0.370xw,0&resize=480:*")

        img = read_img(url)

    join_table, y_table, model_RN50, model_2lr, model_2cr, model_2dr, model_2jr, model_2gr, dog_features,train_breeds_names,train_breeds_features = read_files_and_models()

    image_to_tensor = np.zeros(((1), 224, 224, 3), dtype='float32')

    image_to_tensor[0] = preprocess_input(np.expand_dims(img.copy(), axis=0))
    get_image_features = model_RN50.predict(image_to_tensor)

    pred = ((model_2cr.predict(get_image_features)+model_2dr.predict(get_image_features)+model_2gr.predict(get_image_features)+model_2jr.predict(get_image_features)+model_2lr.predict(get_image_features))/5)

    top_5 = [i[0] for i in sorted(list(zip(list(y_table.columns),pred.tolist()[0])), key = lambda x: x[1], reverse = True)[:5]]
    label = list(y_table.columns)[int(pred.argmax(axis=1))]

    st.title("Predicted label")
    st.header(label.replace("_"," ").title())
    if img_path == "Local":
        show_test_img(name)
    else:
        show_image_pred(url, pred.argmax(axis=1))
    similar_pic = sorted(list(zip(list(train_breeds_names[0]),(cosine_similarity(get_image_features,pd.DataFrame(train_breeds_features))).reshape(-1, 1))), key=lambda x: x[1], reverse = True)[0][0]

    if st.sidebar.checkbox('Do you want the most similar pic in database?'):
        st.header("Most similar picture in database")
        show_image(similar_pic)

    if st.sidebar.checkbox('Most visually similar dogs'):
        st.header("Most visually similar dogs")
        show_rand_image(top_5[1:])
        if st.checkbox("Do you want to see their features?"):
            st.dataframe(find_features_table(top_5[1:]), 2000, 1000)

    if st.sidebar.checkbox('Do you want to see the features of your dog?'):
        st.header("Features of your dog")
        if dog_features.loc[dog_features['label_name'] == label].values[0][1:-1][0] == 0:
            st.text('This dog is not a pet or no information available')
            if st.checkbox('Do you want to see the features of visually similar dogs?'):
                st.dataframe(find_features_table(top_5[1:]), 2000, 1000)
        st.dataframe(find_features_table([label]), 2000, 1000)

    if st.sidebar.checkbox('Do you want to find similar dogs based on its features?'):
        st.header("Most similar dogs based on their features")
        top_5_feat = [x[0] for x in sorted(list(zip(list(dog_features.label_name),(cosine_similarity(dog_features[dog_features.columns[1:-1]])[int(pred.argmax(axis=1))]))), key=lambda x: x[1], reverse = True)[1:5]]
        st.dataframe(find_features_table(top_5_feat), 2000, 1000)
        if st.checkbox('Do you want to see them?'):
            show_rand_image(top_5_feat)
