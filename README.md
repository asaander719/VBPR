# VBPR
VBPR is a modified version with two simple BPR branches for compatibility modeling between a given top and a bottom to be matched. These branches are dedicated to measuring similarity in ID embeddings and visual latent features, employing the cosine function for all similarity prediction. Notably, both tops and bottoms are initialized using the same ID embedding table and a shared latent visual encoder, which is a straightforward MLP with only one layer. We use AUC as the evaluation metric, the goal is simply chose the right bottom between a postive and a negtive target.

In terms of the latent visual encoder's input, we leverage pretrained visual features generated from ResNet. Importantly, we maintain the ResNet frozen without updates during the training of the recommendation model (VBPR). However, there is flexibility in the approach, allowing you to customize the scheme by integrating your own visual feature encoder. This way, you can concatenate it with our recommendation encoder, enabling the generation of results directly from raw visual images. 

## Dataset: IQON3000 
- You may refer to the the public dataset IQON3000 we use in the paper at:
  https://xuemengsong.github.io/fp506-songA.pdf

- Download and unzip the datasets with indexed and pretrained visual feature, then move them to **./dataset/** from: 
  https://drive.google.com/file/d/1gIiLTJ9UJDm8nsdC0Kc9PMKavuSUNDd1/view?usp=sharing

- You may also refer to the raw data and images from GP-BPR :) 
  https://github.com/hanxjing/GP-BPR

- We keep the same data process and setting as GP-BPR, where each sample in train/valid/test csv files is formated as "user_ID, top_ID, pos_bottom_ID, neg_bottom_ID", and we further index them as "user_ID_index, top_ID_index, pos_bottom_ID_index, neg_bottom_ID_index" according to **dataset/IQON3000/data/user_map.json** and **dataset/IQON3000/data/item_map.json**. However, user information is not useful in our model.

## Run VBPR

Change the setting in the config file, then `python run_VBPR.py`.
