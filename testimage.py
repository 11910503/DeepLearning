import predicter
from PIL import Image, ImageTk
import cv2

if __name__ == "__main__":
    predicter = predicter.predicter('weigth/yolov5s.pt')#加载模型
    image=cv2.imread('test_image.jpg')#读取图像

    ans=predicter.runtest2(image)
    print(ans)
