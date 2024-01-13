DATASET = "./data/data.yaml"
EPOCHS = 100
BATCH_SIZE = 12
DEVICE = "cuda"
MODEL_CONFIG = "./yolo/yolov8.yaml"
PRETRAINED_MODEL = "./yolo/yolov8s.pt"
TRAINED_MODEL = "./yolo/license_plate_detector.pt"
THRESHOLD = 64
LANGUAGES = ["en"]
RECORDS = "./records/record.csv"
CHECKPOINTS_FREQ = 100
RECORD_COLUMNS = [
    "video_id", "timestamp", "car_id", "car_box", "plate_box", "transcript", "confidence"
]
