from ultralytics import YOLO
import cv2

class TrafficSignDetector:
    def __init__(self, yolo_model_path='yolov8n.pt'):
        # Puedes usar un modelo preentrenado o uno ajustado a señales de tránsito
        self.model = YOLO(yolo_model_path)

    def detect(self, image):
        """
        Recibe una imagen (numpy array) y retorna una lista de bounding boxes:
        Cada bounding box es (x1, y1, x2, y2, score, class_id)
        """
        results = self.model(image)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                detections.append((int(x1), int(y1), int(x2), int(y2), float(score), class_id))
        return detections

# Ejemplo de uso:
if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    detector = TrafficSignDetector()
    detections = detector.detect(image)
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, f"{class_id}:{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow("Detections", image)
    cv2.waitKey(0)