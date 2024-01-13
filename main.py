import numpy as np
from sort.sort import Sort
import cv2
from util import (
    get_car_detector, get_plate_detector, get_ocr,
    read_plate, process_text, get_car
)
from writer import RecordManager

plate_detector = get_plate_detector()
car_detector = get_car_detector()
reader = get_ocr()


def record(
    video_path="./data/sample.mp4", preview=False,
    output_resize=(960, 540), records_path="./records/record.csv",
    new_record=False
):
    print(f"rendering {video_path}")
    vid = cv2.VideoCapture(video_path)
    vehicle_ids = [2, 3, 5, 7]
    car_tracker = Sort()
    timestamp = 0
    recorder = RecordManager(file_path=records_path, initialize=new_record)

    try:
        while vid.isOpened():
            ret, frame = vid.read()

            cars = car_detector(frame, verbose=False)[0]
            car_boxes = cars.boxes.data.tolist()
            detections = []

            for box in car_boxes:
                x1, y1, x2, y2, score, class_id = box
                detections.append([x1, y1, x2, y2, score])

            tracks = car_tracker.update(np.asarray(detections))
            plates = plate_detector(frame, verbose=False, conf=0.3)[0]

            for box in plates.boxes.xyxy.data.tolist():
                x1, y1, x2, y2 = map(int, box)
                transcript, confidence = read_plate(
                    reader, frame[y1-10:y2+10, x1-10:x2+10])

                if not transcript:
                    continue

                formated_transcript = process_text(transcript)
                if not formated_transcript:
                    continue

                xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                    tracks, x1, y1, x2, y2)

                if not car_id:
                    continue

                recorder.add_record(
                    video_id=video_path,
                    timestamp=timestamp,
                    car_id=car_id,
                    car_box=[xcar1, ycar1, xcar2, ycar2],
                    plate_box=[x1, y1, x2, y2],
                    transcript=formated_transcript,
                    confidence=confidence
                )

                if preview:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                    cv2.putText(frame, formated_transcript, (x1, y1),
                                1, 4, (250, 250, 250), 4)
                    cv2.rectangle(frame, (xcar1, ycar1),
                                  (xcar2, ycar2), (255, 0, 0), 4)

            if preview:
                squeezed_frame = cv2.resize(frame, output_resize)
                cv2.imshow("Traffic", squeezed_frame)

            timestamp += 1
            
            if preview and cv2.waitKey(1) & 255 == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    recorder.save()
    vid.release()
    cv2.destroyAllWindows()
    print(f"rendering is finished with {timestamp} frames processed")


if __name__ == "__main__":
    record()
