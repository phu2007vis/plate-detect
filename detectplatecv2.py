import cv2
import  time
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
from  utils import  *
model = YOLO("best.pt")
image_size = (224,224)
video = cv2.VideoCapture("sample.mp4")

frame = video.read()[1]
frame = cv2.resize(frame,(0,0),fx = 1/4,fy = 1/4)

r = cv2.selectROI("select a roi:",frame)
r = [i*4 for i in r]

transform = transforms.Compose(
    [

        transforms.ToTensor()
    ]
)
i = 0
while True:
    i+=1
    if i%3!=0:
        continue

    t = time.time()

    ret,frame = video.read()
    frame_new = frame[r[1]:r[1]+r[3],r[0]:r[0]+r[2],:]
    frame = cv2.resize(frame_new,image_size)

    frame2 = Image.fromarray(frame)
    torch_image = transform(frame2).unsqueeze(0)
    plates = model(torch_image,verbose = False)[0]

    boxes = plates.boxes.xyxyn
    plates_image = get_plates(frame_new,boxes)
    texts = get_text(plates_image)



    for i,box in enumerate(plates.boxes.xyxy):

        x1,y1,x2,y2 = [int(value) for i,value in enumerate(box.tolist())]
        frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
        text = texts[i]
        cv2.putText(frame,text,(x1,y1),1,2,(0,255,0),2,cv2.LINE_AA)



    if not ret:
        video.release()
        cv2.destroyAllWindows()
        break

    cv2.imshow("frame",frame)
    cv2.waitKey(2)




