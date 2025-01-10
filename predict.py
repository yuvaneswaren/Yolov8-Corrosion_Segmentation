import cv2
from ultralytics import YOLO
img_pth = "test/images/Captura-de-Tela-2023-01-25-a-s-10-32-09_png_jpg.rf.626f986b01572e292640623ee610f81c.jpg"
model = YOLO("runs/segment/train3/weights/best.pt")
results = model(source=img_pth)
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)

