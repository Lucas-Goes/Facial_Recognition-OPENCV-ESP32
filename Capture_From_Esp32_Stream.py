import cv2
import urllib.request
import numpy as np
import dlib

hog_face_detector = dlib.get_frontal_face_detector()
stream = urllib.request.urlopen('http://192.168.100.16:81/stream')
stream_bytes = b''

while True:
    try:
        # decode motion-jpeg bit by bit to jpg
        stream_bytes += stream.read(1024)  # receives the value of the esp32 stream reading in bytes
        first = stream_bytes.find(b'\xff\xd8')  # all the jpg iniciate with \xff\xd8 code
        last = stream_bytes.find(b'\xff\xd9')  # all the jpg finish with \xff\xd8 code
        if first != -1 and last != -1:  # detect first and last code byte to form image
            jpg = stream_bytes[first:last + 2]  # consolidate jpg
            stream_bytes = stream_bytes[last + 2:]  # continuos to next jpg frame
            imagem = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            # imdecode recebe um valor salvo no buffer, np.frombuffer usa a interface do buffer com formato de 8 bits
            # recebe o vetor de bytes salvo na variavel jpg, e retorna convertida em uma imagem colorida
            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # convert current frame to gray
            # apply face detection (hog)
            faces_hog = hog_face_detector(gray, 1)
            for face in faces_hog:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                cv2.rectangle(gray, (x, y), (x + w, y + h), (209, 203, 203), 1)
                cv2.imwrite("face_detected/cap.png", imagem)
                print("imagem salva")

            # Display the resulting frame
            print("stream on")
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:  # important block try except
        continue  # this  is required to deal with error due to blank frames received

# When everything done, release the capture
cv2.destroyAllWindows()
