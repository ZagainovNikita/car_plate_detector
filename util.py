from ultralytics import YOLO
import easyocr as ocr
import pandas as pd
import numpy as np
import cv2
import config


def get_car_detector(model_path=config.PRETRAINED_MODEL):
    return YOLO(model_path)


def get_plate_detector(model_path=config.TRAINED_MODEL):
    return YOLO(model_path)


def get_ocr(languages=config.LANGUAGES):
    return ocr.Reader(lang_list=languages, gpu=True)


def is_digit(char):
    return (48 <= ord(char) <= 57)


def is_letter(char):
    return (57 <= ord(char) <= 90)


def read_plate(reader, snippet, thresh=config.THRESHOLD):
    gray = cv2.cvtColor(snippet, cv2.COLOR_RGB2GRAY)
    _, plate = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    detection = reader.readtext(plate)
    # print(detection)
    if detection:
        text = detection[0][1]
        confidence = detection[0][2]
        return text.upper(), confidence
    return None, None


def process_text(input_plate):
    plate = input_plate.upper()
    forbidden_chars = []
    letter_num = {
        "I": "1",
        "O": "0",
        "S": "5",
        "G": "6",
        "B": "8",
        "J": "3"
    }
    num_letter = {v: k for k, v in letter_num.items()}

    for char in plate:
        if not (is_digit(char) or is_letter(char)):
            forbidden_chars.append(char)
    for char in forbidden_chars:
        plate = plate.replace(char, "")

    if len(plate) != 7:
        return None

    plate = list(plate)

    try:
        for i, char in enumerate(plate):
            if (i <= 1 or i >= 4) and is_digit(char):
                plate[i] = num_letter[char]
            elif (i >= 2 and i <= 3) and is_letter(char):
                plate[i] = letter_num[char]
    except KeyError:
        return None

    return "".join(plate)


def get_car(tracks, x1, y1, x2, y2):
    for track in tracks:
        xcar1, ycar1, xcar2, ycar2, car_id = track
        if xcar1 < x1 and xcar2 > x2 and ycar1 < y1 and ycar2 > y2:
            return int(xcar1), int(ycar1), int(xcar2), int(ycar2), car_id
    return None, None, None, None, None
