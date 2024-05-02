# Objective
Use synthetically generated images to train a set of computer vision models that can detect changes in elevation and azimuth of a device.




# Setup: 3 Easy Steps

## 1 Generate Synthetic Dataset
### See Notebook on Blender Model Generation

##### You need to run the py file for your model inside of blender's "Scripting" tab in a .blend file.  

This is the .txt document under "notebooks"

The .blend files are saved for reference.

There are two dependent folders in data/external/ that need to be added for the script in the .blend files to work, /hdr and /textures.  

HDR's are 100's of MB's and can be sourced from a websearch. I've linked my source in the attribution page.  Textures can be png's or jpegs you want to cover the model with.

Reccommend visiting a tutorial via Zumo-Labs.  Linked in Attribution.

![Screenshot from 2024-05-01 20-42-32](https://github.com/abilokonsky/msds_practicum/assets/62521066/ca005dd1-9173-449a-929a-12cf8554cd90)

## 2 Train Models
### See Elevation and Azimuth Model Training Notebook
##### Special Note: I have a few dozen notebooks.  I only uploaded a few to show concept of how I trained data.



## 3. Deploy
### Run Live Inference.  
200x200 images are what I eventually used for training, however after reflection and some pre-built models not working well with my images, I recommend 224x224 for future developers as this is the imagenet default)



![predictions](https://github.com/abilokonsky/msds_practicum/assets/62521066/9d08cc2d-dc7a-402b-a79d-f29a6fbda702)
