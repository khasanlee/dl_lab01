import cv2
import torch
import requests

try:
    # Load the YOLOv5 model from the Pytorch Hub (https://pytorch.org/hub/)
    yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

    # Load an image on internet
    image = cv2.imread('test.jpg')

    if image is None:
        raise FileNotFoundError("Image not found. Check the file path.")

    # Detect objects on the image using the YOLO
    results = yolov5(image)
    objects = results.xyxyn[0].detach().cpu().numpy()

    # Rescale object locations
    h, w, _ = image.shape
    objects[:, 0:4] = objects[:, 0:4] * [w, h, w, h]

    # Show the image with results
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    for obj in objects:
        if obj[-2] > 0.5:  # More than 0.5 confidence
            tl, br = obj[0:2].astype('int'), obj[2:4].astype('int')
            cv2.rectangle(image, tuple(tl), tuple(br), (0, 0, 255), 2)
            cv2.putText(image, f'{classes[int(obj[-1])]}: {obj[-2]:.2f}', tuple(tl + (-2, -4)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255))

    cv2.imshow('pytorch_yolo', image)
    cv2.waitKey(0)

    # Save the resulting image
    cv2.imwrite('result_with_boxes.jpg', image)
    print("Resulting image saved as 'result_with_boxes.jpg'.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cv2.destroyAllWindows()
