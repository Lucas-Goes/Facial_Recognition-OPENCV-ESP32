# program for recognize images saved from esp32/webcam
# developed by Lucas De Góes Santos for the Bachelor of Computer Engineering
import pickle
import cv2
import dlib
import face_recognition
import imutils
from imutils.face_utils import FaceAligner
import _thread as tr


args = {
    "encodings": "encodings.pickle",
    "image": "face_detected/cap.png",
    "shape_predictor": "shape_predictor_68_face_landmarks.dat"
}

data_frame = pickle.loads(open(args["encodings"], "rb").read())  # load data frames from pickle encodings file

hog_face_detector = dlib.get_frontal_face_detector()


def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _raw_face_locations(img, number_of_times_to_upsample):
    return hog_face_detector(img, number_of_times_to_upsample)


def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def face_locations(img, number_of_times_to_upsample):
    return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in
            _raw_face_locations(img, number_of_times_to_upsample)]


def fn_recognition(img_to_recognition, n):
    print("[INFO] recognizing faces...")

    boxes = face_locations(img_to_recognition, 1)
    encodings = face_recognition.face_encodings(img_to_recognition, boxes)

    names = []

    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data_frame["encodings"], encoding)
        name = "Intruder"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data_frame["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)


    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(img_to_recognition, (left, top), (right, bottom), (158, 173, 62), 1)
        cv2.rectangle(img_to_recognition, (left, top), (left+200, top-25), (158, 173, 62), -1)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(img_to_recognition, name, (left, top-5), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
        cv2.imwrite("face_detected/recognized" + str(n) + ".png", img_to_recognition)
        print("Reconhecido")
        recognized_image = "recognized" + str(n) + ".png"
        return recognized_image


def fn_align_face(face_to_align):
    predictor = dlib.shape_predictor(args["shape_predictor"])
    fa = FaceAligner(predictor, desiredFaceWidth=400, desiredFaceHeight=420)
    #face_aligned = imutils.resize(face_to_align, width=800)
    gray = cv2.cvtColor(face_to_align, cv2.COLOR_BGR2GRAY)
    boxes = hog_face_detector(gray, 2)

    faces_name = []
    faces = 0
    for box in boxes:
        faces += 1
        face_aligned = fa.align(face_to_align, gray, box)
        #cv2.putText(face_aligned, "", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite("face_detected/align" + str(faces) + ".png", face_aligned)
        faces_name.append("align" + str(faces) + ".png")
    return faces_name


def fn_detect_face(img_to_detect):
    gray = cv2.cvtColor(img_to_detect, cv2.COLOR_BGR2GRAY)
    boxes = hog_face_detector(gray, 2)

    count_faces = 1
    for face in boxes:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(img_to_detect, (x, y), (x + w, y + h), (158, 173, 62), 1)  # blue, green, red
        cv2.rectangle(img_to_detect, (x, y), (x + 70, y - 14), (158, 173, 62), -1)  # blue, green, red
        cv2.putText(img_to_detect, "FACE: " + str(count_faces), (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 100), 1)
        count_faces += 1
    cv2.putText(img_to_detect, "Faces detected: ", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img_to_detect, str(count_faces), (145, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow("Faces detected", img_to_detect)
    cv2.waitKey(3000)


def show_image():
    while True:
        try:
            image = cv2.imread(args["image"])
            image2 = cv2.imread(args["image"])

            if image is None:
                print("Could not read input image")
                show_image()

            fn_detect_face(image2)

            call_align = fn_align_face(image)
            print(call_align)

            for filenames in call_align:
                print(filenames)
                n = filenames[5:-4]
                path = {"image": "face_detected/" + str(filenames)}

                face_aligned_saved = cv2.imread(path["image"])

                cv2.imshow("Aligned Image", face_aligned_saved)

                call_recognition = fn_recognition(face_aligned_saved, n)

                path_recognized = {"image_recognized": "face_detected/" + str(call_recognition)}

                face_recognized_saved = cv2.imread(path_recognized["image_recognized"])

                cv2.imshow("Recognized Image", face_recognized_saved)

                cv2.waitKey(2000)
        except:
            continue

show_image()
