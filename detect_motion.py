import cv2
import argparse
import numpy as np
import imutils
import time

def parse_video(videoPath):
    cap = cv2.VideoCapture(0)
    cap.open("http://192.168.0.121:8080/video")
    trace = []
    num_frames = 0
    if(cap.isOpened()==False):
        print("Error opening web stream")
        exit(0)

    cv2.startWindowThread()
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_blur = cv2.GaussianBlur(frame, (5,5), 0)
        lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        print(min(list(b)))
        thresh = cv2.threshold(b, 75, 255, cv2.THRESH_BINARY_INV)[1]
        thresh1 = cv2.dilate(thresh, (3,3), iterations=10)
        cnts = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            (x,y) = ((m["m10"]/float(m["m00"])), (m["m01"]/float(m["m00"])))
            trace.append((x,y))
            for (x,y) in trace:
                cv2.drawContours(frame, [c], -1, (0,255,0), 1)
                cv2.circle(frame, (int(x),int(y)), 7, (0,255,255), -1)
                #cv2.rectangle(frame, (int(x)-10, int(y)-10), (int(x)+10, int(y)+10), (0,255,255), 2)
        num_frames += 1
        if num_frames%30==0:
            end_time = time.time()
            delta = end_time - start_time
            start_time = end_time
            print("FPS: ", str(30/float(delta)))
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF==('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True)
    args = vars(ap.parse_args())
    parse_video(args["video"])