from ultralytics import YOLO
import config

yolo = YOLO(config.MODEL_CONFIG).load(config.PRETRAINED_MODEL)

if __name__ == "__main__":
    results = yolo.train(
        data=config.DATASET,
        epochs=config.EPOCHS,
        batch=config.BATCH_SIZE,
        device=config.DEVICE
    )
