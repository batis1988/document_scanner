#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re
import string
import warnings
warnings.filterwarnings('ignore')

# load NER model 
modelNER = spacy.load('./output/model-best/')


# help function to clean up you text
def cleanText(text: str) -> str:
    """
    Designed for:
    -------------
    removement of  whitespaces and special characters from source text
    Input:
    ------
    str to preprocess
    Output:
    -------
    clean str
    """
    whitespace = string.whitespace
    punctuation = '!#$%&\'()*+/:;<=>?[\\]^`{|}~'
    table_whitespace = str.maketrans('', '', whitespace)
    table_puctuation = str.maketrans('', '', punctuation)
    text = str(text)
    text = text.lower()
    removeWhitespace = text.translate(table_whitespace)
    removePunctuation = removeWhitespace.translate(table_puctuation)
    return str(removePunctuation)

def parser(text, label):
    if label == "PHONE":
        text = text.lower()
        text = re.sub(r'\D', '', text)
    elif label == "EMAIL":
        text = text.lower()
        allow_special_char = "@_.\-"
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text)
    elif label == "WEB":
        text == text.lower()
        allow_special_char = ":/.%#\-"
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text)
    elif label in ("DES", "NAME"):
        text == text.lower()
        text = re.sub(r'[^A-Za-z ]', '', text)
        text = text.title()
    elif label == "ORG":
        text == text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = text.title()
    return text

class GroupGen:
    def __init__(self):
        self.id = 0
        self.text = ''

    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id

# grouper to pass a callable
grp_gen = GroupGen()

# load image
#img = cv2.imread('./data/6.jpg')

def get_predictions(img):
    # extract data with pytesseract
    rawData = pytesseract.image_to_data(img)
    # convert to a pd dataframe
    rawList = list(map(lambda x: x.split('\t'), rawData.split('\n')))
    df = pd.DataFrame(rawList[1:], columns=rawList[0])
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(cleanText)
    df_clean = df.query('text != "" ')
    content = " ".join([word for word in df_clean['text']])
    
    # get predictions
    doc = modelNER(content)
    
    # converting doc to json
    docjson = doc.to_json()
    doc_text = docjson['text']
    
    # creating tokens
    data_tokens = pd.DataFrame(docjson['tokens'])
    data_tokens['text'] = data_tokens[['start', 'end']].apply(
        lambda x: doc_text[x[0]:x[1]], axis=1)
    
    # join table to a df
    dataframe_tokens = pd.merge(data_tokens, 
                  pd.DataFrame(docjson['ents'])[['start', 'label']], 
                  how='left', 
                  on='start')
    dataframe_tokens.fillna('O', inplace=True)
    
    dataframe_tokens.columns = ['id', 'start', 'end', 'token', 'label']
    
    # Bounding box preparation
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
    df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)
    data_info = pd.merge(df_clean, 
             dataframe_tokens[['start', 'token', 'label']], 
             how='inner', 
             on='start')
    
    # Creating BBOX
    bbox = data_info.query("label != 'O' ")
    image = img.copy()
    
    for x, y, w, h, label in bbox[['left', 'top', 'width', 'height', 'label']].values:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 10)
    
    bbox['label'] = bbox['label'].apply(lambda x: x[2:])
    bbox['group'] = bbox['label'].apply(grp_gen.getgroup)
    
    # right and bottom of bbox
    bbox[['left', 'top', 'width', 'height']] = bbox[['left', 'top', 'width', 'height']].astype(int)
    bbox['right'] = bbox['left'] + bbox['width']
    bbox['bottom'] = bbox['top'] + bbox['height']
    
    # tagging : group by groups
    columns_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = bbox[columns_group].groupby(by='group')
    img_tagging = group_tag_img.agg({
        'left': min, 
        'right': max, 
        'top': min, 
        'bottom': max, 
        'label': np.unique, 
        'token': lambda x: " ".join(x)})
    
    # draw a bbox on image
    img_bb = img.copy()
    for l, r, t, b, label, token in img_tagging.values:
        cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(img_bb, str(label), (l, t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    
    # Entities
    info_array = data_info[['token', 'label']].values
    entities = dict(
        NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
    previous = "O"
    
    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]
        text = parser(token, label_tag)
        if bio_tag in ('B', 'I'):
            if previous != label_tag:
                entities[label_tag].append(text)
            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)
                else:
                    if label_tag in ("NAME", "ORG", "DES"):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text
        previous = label_tag

    return content, img_bb, entities
