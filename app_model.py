import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract as pt
import onnxruntime as ort

# pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


INPUT_WIDTH =  640
INPUT_HEIGHT = 640
net = cv2.dnn.readNet('./static/models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype='float32')
    input_image[0:row, 0:col] = image
    
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # print(len(boxes_np))
    # print(len(confidences_np))
    if len(boxes_np)==0:
        return boxes_np, confidences_np, [-1]
    
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()
    
    return boxes_np, confidences_np, index

def extract_text(image,bbox):
    x,y,w,h = bbox
    
    roi = image[y:y+h, x:x+w]
    if 0 in roi.shape:
        return ''
    else:
        roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
        resize_img = cv2.resize(roi_bgr, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)
        gaussian_blur_img = cv2.GaussianBlur(gray, (5, 5), 0)

        # text = pt.image_to_string(magic_color,lang='eng',config='--psm 6')
        text = pt.image_to_string(gaussian_blur_img, lang ='eng',
                config ='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        text = text.strip()
        
        return text

def drawings(image,boxes_np,confidences_np,index):
    if index[0]==-1:
        return image, []

    else: 
        text_list = []
        for ind in index:
            x,y,w,h =  boxes_np[ind]
            bb_conf = confidences_np[ind]
            conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
            license_text = extract_text(image,boxes_np[ind])

            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,255), 10)
            cv2.rectangle(image, (x,y-30), (x+w, y), (0,0,0),-1)
            
            cv2.rectangle(image, (x,y+h), (x+w, y+h+30), (0,0,0),-1)

            cv2.putText(image, conf_text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),3)
            
            cv2.putText(image, license_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),2) # text

            text_list.append(license_text)

        return image, text_list


def test_function_for_bb_detections(img):
    input_image, detections = get_detections(img,net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)

    if index[0]==-1:
        return -1, -1

    text_list = []
    bb_confidences= []
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        license_text = extract_text(img, boxes_np[ind])

        bb_confidences.append(bb_conf)
        text_list.append(license_text)

    return text_list, bb_confidences



def yolo_preds_for_real_time(img):
    input_image, detections = get_detections(img,net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # result_img, text = drawings(img,boxes_np,confidences_np,index)
    return boxes_np, confidences_np, index


def yolo_predictions(img,net):
    input_image, detections = get_detections(img,net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    result_img, text = drawings(img,boxes_np,confidences_np,index)
    return result_img, text


def object_detection(path,filename):
    image = cv2.imread(path)
    image = np.array(image,dtype=np.uint8) 
    result_img, text_list = yolo_predictions(image,net)
    cv2.imwrite('./static/predict/{}'.format(filename),result_img)

    text = ""
    for txt in text_list:
        text = text + ", " + txt

    # print(text[2:])

    return text[2:]
