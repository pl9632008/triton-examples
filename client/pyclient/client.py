
import numpy as np
import cv2
import time
import tritonclient.http as httpclient

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

if __name__ == "__main__":
    client = httpclient.InferenceServerClient(url="localhost:8000")
    raw_image = cv2.imread("zidane.jpg")
    
    print(raw_image)
    print(raw_image.shape)

    a= time.time()

    detection_input = httpclient.InferInput("ensemble_input",raw_image.shape,"UINT8")
    detection_input.set_data_from_numpy(raw_image)

    detection_dims = httpclient.InferInput("ensemble_dims",[3],"INT32")

    dims = np.array(raw_image.shape,dtype=np.int32)
    detection_dims.set_data_from_numpy(dims)

    detection_response = client.infer(model_name="ensemble_model",inputs=[detection_input,detection_dims])

    output_data = detection_response.as_numpy("ensemble_output")
    b=time.time()
    print("time = {:.3f} seconds".format((b-a)))

    for i in output_data:
        x,y,w,h,score,label = int(i[0]),int(i[1]),int(i[2]),int(i[3]),round(i[4],2),int(i[5])
        
        print(x,y,w,h,score,label,sep=" ")
        
        cv2.rectangle(raw_image,(x,y),(x+w,y+h),(0,0,255))
        
        cv2.putText(raw_image,str(score)+"  "+CLASSES[label],(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 255, 0))
        
    cv2.imwrite("result.jpg",raw_image)

