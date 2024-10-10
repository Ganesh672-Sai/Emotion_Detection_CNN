import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

# Load pre-trained model and face detector
face_classifier = cv2.CascadeClassifier(r'C:\Users\mudug\Downloads\Emotion_Detection_CNN-main 2\Emotion_Detection_CNN-main 2\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\mudug\Downloads\Emotion_Detection_CNN-main 2\Emotion_Detection_CNN-main 2\model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Global flag to control the loop for video capture
running = False

# GUI class for Emotion Detector
class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detector")
        self.root.geometry("700x500")
        self.root.configure(bg="lightblue")
        
        # Title label
        self.label_title = tk.Label(self.root, text="Real-time Emotion Detection", font=("Arial", 18, "bold"), bg="lightblue", fg="black")
        self.label_title.pack(pady=20)
        
        # Start button
        self.start_button = tk.Button(self.root, text="Start", font=("Arial", 14), command=self.start_camera, bg="green", fg="white", width=10)
        self.start_button.pack(pady=10)

        # End button
        self.end_button = tk.Button(self.root, text="End", font=("Arial", 14), command=self.stop_camera, bg="red", fg="white", width=10)
        self.end_button.pack(pady=10)

        # Video display frame
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack()

    # Method to start the camera feed
    def start_camera(self):
        global running
        running = True
        thread = threading.Thread(target=self.video_loop)
        thread.start()

    # Method to stop the camera feed
    def stop_camera(self):
        global running
        running = False
        messagebox.showinfo("Emotion Detector", "Stopping the camera...")

    # Video loop for emotion detection
    def video_loop(self):
        cap = cv2.VideoCapture(0)
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    
                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert frame to Tkinter image format
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
            self.video_frame.update()

        cap.release()
        cv2.destroyAllWindows()

# Main method to run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()
