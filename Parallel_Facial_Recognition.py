# program for recognize images saved from esp32 stream script
# developed by Lucas De Góes Santos for the Bachelor of Computer Engineering
import pickle
import cv2
import dlib
import face_recognition

args = {
    "encodings": "encodings.pickle",
    "image": "face_detected/cap.png"
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


def fn_recognition(img_to_recognition):
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
        cv2.rectangle(img_to_recognition, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(img_to_recognition, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        return img_to_recognition


def show_image():
    while True:
        image = cv2.imread(args["image"])

        if image is None:
            print("Could not read input image")
            show_image()

        call_recognition = fn_recognition(image)
        # show the output image
        print("[INFO] Finished ...")
        cv2.imshow("Image_Recog", call_recognition)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        print("stream on")


show_image()
