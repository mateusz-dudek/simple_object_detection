import cv2
from tracker import *

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    else:
        height, width, _ = frame.shape

        # Region of interest
        roi = frame[340:720, 500:800]

        # Object detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for contour in contours:
            # Area calculation to remove elements that are useless

            area = cv2.contourArea(contour)
            if area > 100:
                # Draw rectangle over the detected object
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

                detections.append([x, y, w, h])

        # Object tracking
        boxes_ids = tracker.update(detections)

        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        ## Show video block
        cv2.imshow("DETECTIONS and TRACKING", frame)

        key = cv2.waitKey(20)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
