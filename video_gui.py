import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np

class VideoDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Detection App")
        self.root.geometry("550x550")  # Set window size to 750x750

        self.title_label = tk.Label(root, text="Object Detection on Video")
        self.title_label.pack()

        self.load_button = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_button.pack()

        self.detect_button = tk.Button(root, text="Detect Objects", command=self.detect_objects)
        self.detect_button.pack()

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.video_capture = None
        self.video_width = 0
        self.video_height = 0

        self.model = YOLO("runs/segment/train3/weights/best.pt")

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.video_width = int(self.video_capture.get(3))
            self.video_height = int(self.video_capture.get(4))

    def detect_objects(self):
        if self.video_capture is not None:
            if self.model is not None:
                self.display_video_frame()

    def display_video_frame(self):
        ret, frame = self.video_capture.read()

        if ret:
            results = self.model(source=frame)
            res_plotted = results[0].plot()

            res_bgr = cv2.cvtColor(np.array(res_plotted), cv2.COLOR_RGB2BGR)
            res_photo = ImageTk.PhotoImage(image=Image.fromarray(res_bgr))

            self.video_label.config(image=res_photo)
            self.video_label.image = res_photo

            self.root.after(1, self.display_video_frame)  # Call the function again after a delay

    def display_message(self, message):
        self.video_label.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoDetectionApp(root)
    root.mainloop()