import mediapipe as mp
import numpy as np

class SelfiSegmentation():

    def __init__(self, model=1):

        self.model = model  # Initialize the model type
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(model_selection=self.model)  # Initialize the Selfie Segmentation model from MediaPipe

    def removeBG(self, frame, imgBg, cutThreshold=0.1):

        results = self.selfieSegmentation.process(frame)  # Process the image using Selfie Segmentation model
        condition = np.stack( (results.segmentation_mask, ), axis=-1) > cutThreshold  # Boolean condition based on segmentation mask and cut threshold

        if isinstance(imgBg, tuple):
            _imgBg = np.zeros(frame.shape, dtype=np.uint8)  # Create a colored background image
            _imgBg[:] = imgBg  # Set the color of the background
            imgOut = np.where(condition, frame, _imgBg)  # Replace pixels based on the condition
        else:
            imgOut = np.where(condition, frame, imgBg)  # Replace pixels based on the condition

        return imgOut
