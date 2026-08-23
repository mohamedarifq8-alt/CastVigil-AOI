import cv2
import numpy as np
import customtkinter as ctk
import onnxruntime as ort
from PIL import Image
from tkinter import filedialog

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FactoryDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AOI Quality Control Dashboard - Production Line")
        self.geometry("950x650")

        self.MODEL_PATH = "casting_model.onnx"
        self.CLASS_NAMES = ["Defective", "OK"]
        
        self.session = ort.InferenceSession(self.MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name

        self.cap = cv2.VideoCapture(0)
        self.is_camera_active = True

        # Layout configuration
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Video/Image display frame
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        # Control Panel Frame
        self.control_panel = ctk.CTkFrame(self)
        self.control_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.control_panel, text="Control Panel", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=15)

        self.status_box = ctk.CTkLabel(
            self.control_panel, 
            text="Initializing...", 
            font=("Arial", 16, "bold"),
            fg_color="gray", 
            corner_radius=8,
            height=50
        )
        self.status_box.pack(fill="x", padx=15, pady=10)

        self.confidence_label = ctk.CTkLabel(self.control_panel, text="Confidence: --%", font=("Arial", 14))
        self.confidence_label.pack(pady=5)

        self.plc_signal_label = ctk.CTkLabel(self.control_panel, text="PLC Signal: Idle", text_color="gray", font=("Arial", 12))
        self.plc_signal_label.pack(pady=10)

        # Upload Test Image Button
        self.upload_btn = ctk.CTkButton(
            self.control_panel, 
            text="📁 Upload & Test Image", 
            command=self.upload_and_test_image,
            font=("Arial", 13, "bold"),
            fg_color="#3c40c6",
            hover_color="#575fcf"
        )
        self.upload_btn.pack(pady=10, padx=15, fill="x")

        # Resume Live Camera Stream Button
        self.resume_btn = ctk.CTkButton(
            self.control_panel, 
            text="🎥 Resume Live Camera", 
            command=self.resume_camera,
            font=("Arial", 13, "bold"),
            fg_color="#05c46b",
            hover_color="#0be881"
        )
        self.resume_btn.pack(pady=5, padx=15, fill="x")

        self.update_frame()

    def preprocess_image(self, img_rgb):
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = img_resized.astype(np.float32)
        return np.expand_dims(img_array, axis=0)

    def process_prediction(self, probabilities):
        predicted_idx = np.argmax(probabilities)
        predicted_label = self.CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx] * 100

        if predicted_label == "Defective":
            status = "DEFECTIVE ❌"
            color = "#ff4757"
            plc = "🚨 [PLC Signal: 1] Trigger Ejector Arm!"
        else:
            status = "OK (PASSED) 🟢"
            color = "#2ed573"
            plc = "🟢 [PLC Signal: 0] Conveyor Normal"

        self.status_box.configure(text=status, fg_color=color)
        self.confidence_label.configure(text=f"Confidence: {confidence:.2f}%")
        self.plc_signal_label.configure(text=plc)

    def update_frame(self):
        if self.is_camera_active:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = self.preprocess_image(frame_rgb)
                
                outputs = self.session.run(None, {self.input_name: processed})
                probabilities = outputs[0][0]
                self.process_prediction(probabilities)

                img_pil = Image.fromarray(frame_rgb)
                img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(540, 400))
                self.video_label.configure(image=img_ctk)

            self.after(30, self.update_frame)

    def upload_and_test_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.is_camera_active = False # Pause video stream
            
            image_pil = Image.open(file_path).convert("RGB")
            img_rgb = np.array(image_pil)
            
            processed = self.preprocess_image(img_rgb)
            outputs = self.session.run(None, {self.input_name: processed})
            probabilities = outputs[0][0]
            
            self.process_prediction(probabilities)

            img_ctk = ctk.CTkImage(light_image=image_pil, dark_image=image_pil, size=(540, 400))
            self.video_label.configure(image=img_ctk)

    def resume_camera(self):
        if not self.is_camera_active:
            self.is_camera_active = True
            self.update_frame()

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = FactoryDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()