import cv2
import dlib

count_faces = 0
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_hog = hog_face_detector(gray, 1)

    for face in faces_hog:
        print(faces_hog)
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        print(x)
        print(y)
        print(w)
        print(h)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (209, 203, 203), 1)
        cv2.imwrite("face_detected/cap.png", frame)
        print("imagem salva")

    # Display the resulting frame
    print("stream on")
    cv2.imshow('frame', gray)
    cv2.moveWindow("frame", 150, 100)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
