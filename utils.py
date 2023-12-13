
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cpu'
detector = Predictor(config)

def get_plates(image,boxes,format = "xyxyn"):


    image_size = image.shape[::-1][1:]
    image = cv2.GaussianBlur(image,ksize=(5,5),sigmaX=0)

    plates = []
    if format == 'xyxyn':
        for box in boxes:
            x1,y1,x2,y2 = [int(value*image_size[i%2]) for i,value in enumerate(box)]
            plate = image[y1:y2,x1:x2,:]
            plates.append(plate)

    return plates

def get_text(plates):

    texts = []
    for plate in plates:

        text = detector.predict(plate)
        texts.append(text)

    return texts


