import predicter
from PIL import Image, ImageTk
import cv2

if __name__ == "__main__":
    #predicter = predicter.predicter('weigth/yolov5s.pt')#加载模型
    predicter_cls=predicter.predicter_cls('weigth/sandwich_cls.pt')
    image=cv2.imread('testimage_cls.bmp')#读取图像

    #ans=predicter.runtest2(image)
    ans=predicter_cls.run(image)

    ans=predicter_cls.run2('testimage_cls.bmp')
    print(ans)
