import cv2
from ultralytics import YOLO
from service.audio_service import audioService

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path

    def load(self):
        return YOLO(self.model_path)

class Preprocessor:
    def preprocess(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        return image  

class Postprocessor:
    def __init__(self, class_names):
        self.class_names = class_names

    def postprocess(self, results, image):
        annotated = image.copy()
        detected_classes = set()

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = self.class_names.get(int(class_id), "unknown")
                confidence = float(score)
                detected_classes.add(label.lower())

            
                color = (0, 255, 0)

                cv2.rectangle(annotated, (x1, y1), (x2, y2),color,  2)
                cv2.putText(
                    annotated,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        return annotated, list(detected_classes)

class LogicController:
    def __init__(self,beep):
        self.beep = beep
        pass
    def run_logic(self, detected_classes):
        if any(label in "helmet" for label in detected_classes):
            self.beep.play() 
            return "Helmet Detected"
        elif any(label in "no_helmet" for label in detected_classes):
            return "NO helmet detected"
        else:
            return "No helmet detected"

class HelmetDetectionPipeline:
    def __init__(self, model_path, conf_thresh=0.25):
        self.model_loader = ModelLoader(model_path)
        self.model = self.model_loader.load()
        self.conf_thresh = conf_thresh

        self.class_names = {int(k): v for k, v in self.model.names.items()}

        self.preprocessor = Preprocessor()
        self.postprocessor = Postprocessor(self.class_names)
        self.beep = audioService()
        self.logic_controller = LogicController(self.beep)

    def detect(self, frame):
        processed = self.preprocessor.preprocess(frame)
        results = self.model.predict(processed, imgsz=640, conf=self.conf_thresh, verbose=False)
        annotated, detected_classes = self.postprocessor.postprocess(results, processed)

        print("Detected classes:", detected_classes)

        message = self.logic_controller.run_logic(detected_classes)
        return annotated, message

