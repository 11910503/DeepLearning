import tk
import cv2
from PIL import Image, ImageTk
import predicter

class CameraApp:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("app")
        self.predicter=predicter.predicter('weigth/yolov5s.pt')



        self.live_image_label = tk.Label(window)
        self.live_image_label.pack()



        self.captured_page = CapturedPage(window)


        self.video_source = video_source
        self.video_capture  = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), framerate=(fraction)10/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
        if not self.video_capture.isOpened():
            raise ValueError("error", video_source)


        self.width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)


        self.update_live_image()

    def update_live_image(self):

        _, frame = self.video_capture.read()


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        image = Image.fromarray(frame_rgb)
        image_path = "captured_image.jpg"
        image_fiting_path="test_image.jpg"
        image.save(image_path)

        self.predicter.run(image_path,image_fiting_path)
        image2 = Image.open(image_fiting_path)


        image2 = image2.resize((int(self.window.winfo_screenwidth()), int(self.window.winfo_screenheight())))


        photo = ImageTk.PhotoImage(image2)


        self.live_image_label.configure(image=photo)
        self.live_image_label.image = photo


        self.window.after(10, self.update_live_image)

    def capture_image(self):

        _, frame = self.video_capture.read()


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        image = Image.fromarray(frame_rgb)


        image_path = "captured_image.jpg"
        image.save(image_path)



        self.captured_page.update_image(image_path)

class CapturedPage:
    def __init__(self, window):
        self.window = window


        self.image_label = tk.Label(window)
        self.image_label.pack()

    def update_image(self, image_path):

        image = Image.open(image_path)


        image = image.resize((int(self.window.winfo_screenwidth()), int(self.window.winfo_screenheight())))


        photo = ImageTk.PhotoImage(image)


        self.image_label.configure(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":

    root = tk.Tk()


    app = CameraApp(root)


    root.mainloop()
