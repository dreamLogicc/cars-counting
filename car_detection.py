import cv2
import cvlib as cv
import numpy as np
from cvlib.object_detection import draw_bbox
import imageio.v3 as iio


def play(bytes):

    def get_bbox_center(bbox):
        x,y,w,h = bbox
        return (int((x+w)/2), int((y+h)/2))

    frames_count = 0
    cars_count = 0

    for frame in iio.imiter(bytes, format_hint=".mp4"):

        frames_count = frames_count + 1
        if frames_count % 6 != 0:
            continue

        frame_temp = cv2.resize(frame, (1000,600))

        bbox, labels, conf = cv.detect_common_objects(frame_temp)
        frame_temp = draw_bbox(frame_temp, bbox=bbox, labels=labels, confidence=conf)
        cv2.line(img=frame_temp, pt1=(0, 500), pt2=(1000, 500),
                color=(0, 0, 255), lineType=8, thickness=5)

        centers = [get_bbox_center(bb) for bb in bbox]

        for (x,y) in centers:
            if y > 490:
                cars_count +=1
                centers.remove((x,y))

        cv2.putText(frame_temp, f'cars: {cars_count}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_4)


        cv2.imshow('Video', cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) == 27:  # close on escape
            break

    cv2.destroyAllWindows()