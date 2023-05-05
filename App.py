import torch
import numpy as np
import cv2
import pafy
import time
import streamlit as st
from PIL import Image
import base64
from time import perf_counter

st.set_page_config(
        page_title="Tata motors machine parts detector",
        page_icon="Tata-Logo.jpg",
        layout="wide"
    )



page_bg_img = f"""
<style>
.main {{
background-image:url("https://thumbs.dreamstime.com/b/abstract-eye-background-vision-concept-d-rendering-105098094.jpg");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
background-contrast: 5;

}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
class Detection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        
        self.capture_index = capture_index
        self.model= self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
            
        return labels,cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self,results,frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (255,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                bgr = (255,0,0)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                
                
        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        
        
        st.title("\U0001F4F8 :green[Lets Detect the Machine parts]")
        st.subheader(":blue[Empower your maintenance team with machine part detection technology.]")
        st.sidebar.subheader("Choose your option")
        run = False
        stop = False
        camera = None
        if st.sidebar.checkbox('Using Camera',key="camera is used"):
            run = st.button('Start Webcam',key="start")
            stop = st.button('Stop Webcam',key="stop")
            
        
          
                  
        else:
            if st.sidebar.checkbox('Upload the images',key="file"):
               uploaded_file = st.file_uploader(":red[_**CHOOSE A FILE**_]",type=['jpg','png','jpeg'])
               if uploaded_file is not None:
                st.subheader("\U0001F389\U0001F389:green[File uploaded successfully!]")
                our_image = Image.open(uploaded_file)
                #st.image(our_image)
                img = np.array(our_image)
                
                labels, cord = self.score_frame(img)
                
                results = (labels, cord)
                labeled_img = self.plot_boxes(results, img)
                st.image(labeled_img)
                
                
                
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        
         
        while run:
            start_time = time.perf_counter()
            _, frame = camera.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('camera',frame)
            
           
            FRAME_WINDOW.image(frame)
        if stop:
            camera.release()
            run = False





        
        
# Create a new object and execute.
detector = Detection(capture_index=0, model_name="best10.pt")
detector()


