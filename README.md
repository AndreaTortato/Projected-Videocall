
In this project, we will study how video calls can be performed to look more realistic and immersive, instead of simple 2D videos displayed on a surface, by using projected images and particular concepts of image and video processing. 
Will study what are some of the techniques to achieve a realistic projection focusing on background removal techniques, subject detection algorithms and how to correct errors in the projection itself.

The project is divided as follows:

#MAIN
Starts the video capute object to record frames and calls other functions to apply keystoning, background removal and face detection algorithms

# KEYSTONE
Used to tranform a frame by warping it based on the position of 4 points of the frame which will constitute as the corner points of the tranformed frame

# SEGMENTATION
Background removal using Mediapipe libary mp.selfiesegmentation only display the subject (a person)

# DETECTIONFACE
Detencing faces in the image using OpenCV cascade filter classifier function

# BGSUBTRACTION
Raw implemetation of the backgournd removal based on theorical principles only, no temporal consistency, for demointration purpose only
