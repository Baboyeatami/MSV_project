import cv2
import tkinter as tk
from tkinter import Label, Button
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import os
from datetime import datetime

# Load YOLOv8 model
MODEL_PATH = "msv_best.pt"  # Ensure this path is correct
model = YOLO(MODEL_PATH)

# Create output folder
os.makedirs("inference_test", exist_ok=True)

# Initialize Tkinter
root = tk.Tk()
root.title("YOLOv8 MSV Detection")
root.geometry("1280x850")  # Full HD layout

# Capture webcam
cap = cv2.VideoCapture(0)

# ========== Layout Setup ==========
# Top frame (for previews)
top_frame = tk.Frame(root)
top_frame.pack(fill=tk.BOTH, expand=True)

# Bottom frame (for button)
bottom_frame = tk.Frame(root, height=200)
bottom_frame.pack(fill=tk.X)

# Left frame for live camera preview
left_frame = tk.Frame(top_frame, width=750, height=750)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Right frame for inference result
right_frame = tk.Frame(top_frame, width=750, height=750)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Label widgets for camera and inference previews
camera_label = Label(left_frame)
camera_label.pack(expand=True)

inference_label = Label(right_frame)
inference_label.pack(expand=True)

# Capture button
capture_button = Button(
    bottom_frame,
    text="Capture & Inference",
    command=lambda: capture_and_infer(),
    font=("Arial", 35),
    padx=20,
    pady=5
)
capture_button.pack(pady=10)

# Global to store latest frame
latest_frame = None

# ========== Video Loop ==========
def update_video():
    global latest_frame
    ret, frame = cap.read()
    if ret:
        latest_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2.resize(rgb, (750, 750)))
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    root.after(10, update_video)

# ========== Inference & Save ==========
def capture_and_infer():
    global latest_frame
    if latest_frame is None:
        return

    frame = latest_frame.copy()
    results = model(frame)

    # Process detections
    for result in results:
        detected_img = frame.copy()
        count = len(result.boxes)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()  # Confidence score

            # Set the label to "Maize Streak Virus" (ignoring model's default class names)
            label = f"Maize Streak Virus {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detected_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Annotate number of objects detected
        cv2.putText(
            detected_img,
            f"Maize Streak Virus: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"inference_test/inference_{timestamp}.jpg"
        cv2.imwrite(save_path, detected_img)

        # Show in inference panel
        img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(cv2.resize(img_rgb, (750, 750)))
        imgtk = ImageTk.PhotoImage(image=img_pil)
        inference_label.imgtk = imgtk
        inference_label.configure(image=imgtk)

def on_closing():
    print("Closing the app safely...")
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Handle window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start loop
update_video()
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
