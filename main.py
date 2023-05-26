import tkinter as tk
import cv2
from PIL import Image, ImageTk
import predicter

class CameraApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("摄像头应用")
        self.predicter=predicter.predicter('weigth/yolov5s.pt')


        # 创建一个标签，用于显示实时图像
        self.live_image_label = tk.Label(window)
        self.live_image_label.pack()


        # 创建一个新的页面，用于显示截图
        self.captured_page = CapturedPage(window)

        # 打开视频源
        self.video_source = video_source
        self.video_capture  = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), framerate=(fraction)10/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
        if not self.video_capture.isOpened():
            raise ValueError("无法打开视频源", video_source)

        # 获取视频的宽度和高度
        self.width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 更新实时图像
        self.update_live_image()

    def update_live_image(self):
        # 从视频源读取一帧图像
        _, frame = self.video_capture.read()

        # 将图像从BGR颜色空间转换为RGB颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将图像转换为PIL图像
        image = Image.fromarray(frame_rgb)
        image_path = "captured_image.jpg"
        image_fiting_path="test_image.jpg"
        image.save(image_path)
        #保存图像
        self.predicter.run(image_path,image_fiting_path)
        image2 = Image.open(image_fiting_path)

        # 将图像调整到适当的大小
        image2 = image2.resize((int(self.window.winfo_screenwidth()), int(self.window.winfo_screenheight())))

        # 将PIL图像转换为Tkinter图像
        photo = ImageTk.PhotoImage(image2)

        # 在标签中显示实时图像
        self.live_image_label.configure(image=photo)
        self.live_image_label.image = photo

        # 每隔10毫秒更新实时图像
        self.window.after(10, self.update_live_image)

    def capture_image(self):
        # 从视频源读取当前帧图像
        _, frame = self.video_capture.read()

        # 将图像从BGR颜色空间转换为RGB颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将图像转换为PIL图像
        image = Image.fromarray(frame_rgb)

        # 保存图像
        image_path = "captured_image.jpg"
        image.save(image_path)
        print(f"已截取并保存图像：{image_path}")

        # 更新截图页面的图像
        self.captured_page.update_image(image_path)

class CapturedPage:
    def __init__(self, window):
        self.window = window

        # 创建一个标签，用于显示截图
        self.image_label = tk.Label(window)
        self.image_label.pack()

    def update_image(self, image_path):
        # 读取保存的图像
        image = Image.open(image_path)

        # 将图像调整到适当的大小
        image = image.resize((int(self.window.winfo_screenwidth()), int(self.window.winfo_screenheight())))

        # 将PIL图像转换为Tkinter图像
        photo = ImageTk.PhotoImage(image)

        # 在标签中显示截图
        self.image_label.configure(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    # 创建一个Tkinter窗口
    root = tk.Tk()

    # 创建一个摄像头应用实例
    app = CameraApp(root)

    # 运行Tkinter事件循环
    root.mainloop()
