import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

known_face_encodings = []
known_face_names = []

def init_camera():
    global cap, animation_label
    cap = cv2.VideoCapture(0)
    animation_label.config(text="Scanning face...")
    main_window.after(100, recognize_face)

def recognize_face():
    global cap, animation_label, known_face_encodings, known_face_names

    ret, frame = cap.read()
    if not ret:
        cap.release()
        animation_label.config(text="Error: Camera not found")
        return

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_index, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (face_location[3] * 4, face_location[0] * 4),
                      (face_location[1] * 4, face_location[2] * 4), (0, 255, 0), 2)
        cv2.putText(frame, name, (face_location[3] * 4, face_location[2] * 4 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

    if name == "Unknown":
        message = messagebox.askquestion("New Face Detected", "Do you want to save this face?")
        if message == "yes":
            name = simpledialog.askstring("Enter Name", "Please enter the name:")
            if name:
                known_face_names.append(name)
                known_face_encodings.append(face_encodings[face_index])

    animation_label.after(100, recognize_face)

import tkinter as tk
import cv2
import face_recognition
import numpy as np
import tkinter.messagebox as messagebox
from tkinter import simpledialog
from PIL import Image, ImageTk

main_window = tk.Tk()
main_window.title("Face Recognition System")
main_window.geometry("640x480")

panel = tk.Label(main_window)
panel.pack(side="bottom", fill="both", expand="yes")

animation_label = tk.Label(main_window, text="")
animation_label.pack(side="top", fill="x")

init_camera()

main_window.mainloop()

cap.release()
cv2.destroyAllWindows()