import streamlit as st
import torch
import torchvision
from PIL import Image
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords, plot_one_box
from yolov5.utils.torch_utils import select_device

st.title("YOLOv5 Object Detection")
st.write("Upload an image and detect objects!")

file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

@st.cache(allow_output_mutation=True)
def load_model():
    device = select_device('')
    model = attempt_load('yolov5s.pt', map_location=device)
    return model

@st.cache(allow_output_mutation=True)
def detect_objects(image):
    model = load_model()
    device = select_device('')
    img = Image.open(image)
    img = torchvision.transforms.ToTensor()(img).to(device)
    img = img.unsqueeze(0)
    results = model(img)[0]
    results = non_max_suppression(results, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False)
    return results[0]

def show_image(image, results):
    boxes = scale_coords(image.shape[1:], results[:, :4], image.shape[:2]).round()
    for box in boxes:
        x1, y1, x2, y2 = box
        color = (255, 0, 0)
        thickness = 2
        plot_one_box(box, image, color=color, label=None, line_thickness=thickness)
    st.image(image, caption='Detected Objects', use_column_width=True)

def main():
    if file is not None:
        image = file.read()
        image = Image.open(file)
        image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        results = detect_objects(image)
        show_image(image[0].permute(1, 2, 0).cpu().numpy(), results)

if __name__ == '__main__':
    main()
