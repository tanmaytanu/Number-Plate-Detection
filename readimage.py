import cv2
import numpy as np
import easyocr  # EasyOCR library

# Initialize the cascade and EasyOCR reader
plat_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
reader = easyocr.Reader(['en'])  # Initialize EasyOCR with English language

# Load the image
img = cv2.imread('Data/img6.jpg')

# Detect plates
plates = plat_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))

for (x, y, w, h) in plates:
    # Draw rectangle around detected plate
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, 'License Plate', org=(x - 3, y - 3), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                color=(0, 0, 255), thickness=1, fontScale=0.6)
    
    # Crop the license plate region
    plate_image = img[y:y + h, x:x + w]
    
    # Convert cropped plate image to grayscale for OCR
    plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the cropped license plate
    result = reader.readtext(plate_gray)

    # Display OCR result on the image
    for res in result:
        text = res[1]
        print("Detected Number Plate Text:", text)
        cv2.putText(img, text, (x, y + h + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 255, 0), thickness=2)

# Display the output image
cv2.imshow('Detected Plates', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
