# Objective
Using synthetically trained computer vision to detect changes in the set angle of an adjustable device.
![image](https://github.com/abilokonsky/msds_practicum/assets/62521066/51a88b7b-e448-43e5-9d9b-3cc7a5aaedf4)

# Background
The intention of this project is to use computer vision to understand the implications of the geometric orientation of objects in the real world.  This can be critically important for machine decisions regarding safety and understanding of the environment.  By enabling and deploying such decision making at the edge, IoT devices can more rapidly solve life threatening events.  

Traditional computer vision libraries are excellent at identifying and segmenting a number of  classified objects from a scene, however the goal of this project is to use synthetic data to train a computer vision model on a specific, dynamic object.  Whether specific changes of the segmented object's orientation can be ascertained and to understand to what level the computer vision model can interpolate previously untrained angles from new data.

# Methodology

Build a .stl of a swing-arm object with adjustable angles and 3d print it

Synthetically generate images of the .stl at every 5 degree orientation

Train the model using a to-be-determined system.

Deploy the model to run in an edge configuration using an NVIDIA Orin Nano and a MIDI PCI camera in a lab environment.
  Measure the accuracy that model can achieve at determining the object's elevation angle and orientation.

Deploy the model to run in an edge configuration using an NVIDIA Orin Nano and a MIDI PCI camera in a cluttered "real-world" environment.
  Measure the accuracy that model can achieve at determining the object's angle and orientation.

# Lessons Learned

