import cv2
import sys
import os
import easyocr
import numpy as np
from PIL import Image
import time

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()
    return cap

def initialize_ocr():
    try:
        print("Initializing EasyOCR (this may take a moment)...")
        return easyocr.Reader(['en'])  # Initialize for English
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        sys.exit()

def process_plate_text(reader, img_roi):
    try:
        # Convert to RGB for EasyOCR
        img_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image
        img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)
        
        # Apply thresholding
        _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Perform OCR
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
    # Check if the Haar cascade file exists
    harcascade = "haarcascade_russian_plate_number.xml"
    if not os.path.exists(harcascade):
        print(f"Error: Cannot find {harcascade}")
        sys.exit()

    # Initialize camera and OCR
    cap = initialize_camera()
    cap.set(3, 640)  # width
    cap.set(4, 480)  # height
    reader = initialize_ocr()

    min_area = 500
    count = 0
    last_ocr_time = 0
    OCR_COOLDOWN = 0.5  # Seconds between OCR attempts

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Error: Failed to grab frame")
                break

            plate_cascade = cv2.CascadeClassifier(harcascade)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

            for (x,y,w,h) in plates:
                area = w * h
                if area > min_area:
                    # Draw rectangle around plate
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(img, "Number Plate", (x,y-5), 
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                    # Extract ROI
                    img_roi = img[y: y+h, x:x+w]
                    
                    # Perform OCR with cooldown
                    current_time = time.time()
                    if current_time - last_ocr_time >= OCR_COOLDOWN:
                        plate_text, confidence = process_plate_text(reader, img_roi)
                        if plate_text:
                            # Display the detected text and confidence on the main image
                            text_display = f"{plate_text} ({confidence:.2f})"
                            cv2.putText(img, text_display, (x, y+h+30),
                                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
                            # Display the detected text on the ROI
                            cv2.putText(img_roi, plate_text, (5, 25),
                                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
                        last_ocr_time = current_time

                    cv2.imshow("ROI", img_roi)

            cv2.imshow("Result", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and 'img_roi' in locals():
                # Create plates directory if it doesn't exist
                if not os.path.exists('plates'):
                    os.makedirs('plates')
                    
                # Save the plate image with detected text
                plate_text, _ = process_plate_text(reader, img_roi)
                filename = f"plates/scaned_img_{count}_{plate_text if plate_text else 'unknown'}.jpg"
                cv2.imwrite(filename, img_roi)
                
                # Display save confirmation
                cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
                cv2.putText(img, "Plate Saved", (150, 265), 
                           cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                cv2.imshow("Results", img)
                cv2.waitKey(500)
                count += 1

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()