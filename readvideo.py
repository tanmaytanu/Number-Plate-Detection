import cv2
import numpy as np
import easyocr
import time
import tkinter as tk
from tkinter import ttk

class VideoPlayerControls:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.duration = self.total_frames / self.fps
        
        # Control flags
        self.is_playing = True
        self.current_frame = 0
        
        # Create control window
        self.setup_control_window()
        
    def setup_control_window(self):
        self.root = tk.Tk()
        self.root.title("Video Controls")
        
        # Play/Pause button
        self.play_pause_btn = ttk.Button(self.root, text="Pause", command=self.toggle_play_pause)
        self.play_pause_btn.pack(pady=5)
        
        # Timeline slider
        self.timeline = ttk.Scale(self.root, from_=0, to=self.total_frames,
                                orient='horizontal', length=400,
                                command=self.on_timeline_change)
        self.timeline.pack(pady=5, padx=10, fill='x')
        
        # Time label
        self.time_label = ttk.Label(self.root, text="0:00 / 0:00")
        self.time_label.pack(pady=5)
        
    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        self.play_pause_btn.config(text="Pause" if self.is_playing else "Play")
        
    def on_timeline_change(self, value):
        frame_no = int(float(value))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.current_frame = frame_no
        
    def update_timeline(self):
        current_time = self.current_frame / self.fps
        total_time = self.duration
        self.timeline.set(self.current_frame)
        self.time_label.config(text=f"{int(current_time//60)}:{int(current_time%60):02d} / "
                                  f"{int(total_time//60)}:{int(total_time%60):02d}")
        self.root.update()
        
    def close(self):
        self.root.destroy()

def initialize_ocr():
    print("Initializing EasyOCR (this may take a moment)...")
    return easyocr.Reader(['en'])

def process_plate_text(reader, img_roi):
    try:
        img_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)
        _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        results = reader.readtext(img_thresh)
        
        if results:
            text = " ".join([result[1] for result in results])
            confidence = np.mean([result[2] for result in results])
            return text, confidence
        return None, None
    except Exception as e:
        print(f"OCR Error: {e}")
        return None, None

def main():
    # Initialize OCR and video player
    reader = initialize_ocr()
    video_path = 'D:\4-1\Lab\DIP\number plate detect\Data\vid.mp4'
    player = VideoPlayerControls(video_path)
    plat_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    
    last_ocr_time = 0
    OCR_COOLDOWN = 0.5
    
    try:
        while True:
            if player.is_playing:
                ret, frame = player.video.read()
                if not ret:
                    # Loop back to start
                    player.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    player.current_frame = 0
                    continue
                
                player.current_frame += 1
                
                # Process frame
                gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                plates = plat_detector.detectMultiScale(gray_video, scaleFactor=1.2, 
                                                      minNeighbors=5, minSize=(25,25))
                
                current_time = time.time()
                
                for (x,y,w,h) in plates:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                    plate_roi = frame[y:y+h, x:x+w]
                    
                    if current_time - last_ocr_time >= OCR_COOLDOWN:
                        plate_text, confidence = process_plate_text(reader, plate_roi)
                        if plate_text:
                            text_display = f"{plate_text} ({confidence:.2f})"
                            cv2.putText(frame, text_display, (x, y-10),
                                      cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "License Plate", (x, y-10),
                                      cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
                        last_ocr_time = current_time
                    
                    cv2.imshow('Plate ROI', plate_roi)
                
                # Display frame counter
                cv2.putText(frame, f"Frame: {player.current_frame}/{player.total_frames}",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Video', frame)
            
            # Update timeline
            player.update_timeline()
            
            # Handle keyboard input
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                player.toggle_play_pause()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        player.close()
        player.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()