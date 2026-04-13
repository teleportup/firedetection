import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

class ConsensusEnsemble:

    def __init__(self, iou_threshold=0.4, conf_threshold=0.4):

        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.num_models = 3

    def _calculate_iou(self, boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        if interArea == 0:

            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)

    def _get_avg_box(self, cluster):

        n = len(cluster)

        avg_xmin = sum(p[0] for p in cluster) / n
        avg_ymin = sum(p[1] for p in cluster) / n
        avg_xmax = sum(p[2] for p in cluster) / n
        avg_ymax = sum(p[3] for p in cluster) / n

        return [avg_xmin, avg_ymin, avg_xmax, avg_ymax]

    def process(self, preds1, preds2, preds3):

        # Prediction Pooling
        all_preds = []

        for preds in [preds1, preds2, preds3]:

            for p in preds:

                # Внутренний порог для отсечения откровенного мусора до усреднения
                if p[4] >= 0.1: 

                    all_preds.append(p)

        # Spatial Clustering
        clusters = []

        for p in all_preds:

            matched = False

            for cluster in clusters:

                if self._calculate_iou(p[:4], self._get_avg_box(cluster)) >= self.iou_threshold:

                    cluster.append(p)
                    matched = True

                    break

            if not matched:
                clusters.append([p])

        # Fusion & Consensus
        final_detections = []

        for cluster in clusters:

            avg_box = self._get_avg_box(cluster)
            sum_conf = sum(p[4] for p in cluster)
            consensus_conf = sum_conf / self.num_models

            if consensus_conf >= self.conf_threshold:

                final_detections.append([*avg_box, consensus_conf])

        return final_detections

class EnsembleFireSystem:

    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Ensemble Config")
        self.root.geometry("300x150")
        
        self.source = "0"
        self.disp_w = 1920
        self.disp_h = 1080
        
        # Models
        self.model1 = YOLO("Models/runs/detect/smoke/smoke_baseline/weights/best.pt") 
        self.model2 = YOLO("Models/runs/detect/fire/fire_baseline/weights/best.pt")   
        self.model3 = YOLO("Models/runs/detect/model3/model3_baseline/weights/best.pt")
        
        # Thresholds (conf - risk of fire)
        self.ensemble = ConsensusEnsemble(iou_threshold=0.4, conf_threshold=0.2)
        
        # UI
        tk.Button(self.root, text="Select Video", command=self.set_file).pack(pady=5)
        tk.Button(self.root, text="Use Stream", command=self.set_stream).pack(pady=5)
        tk.Button(self.root, text="START", bg="lightgreen", command=self.run_inference).pack(pady=15)

    def set_file(self):

        self.source = filedialog.askopenfilename()

    def set_stream(self):

        self.source = "0"

    def _extract_boxes(self, results):

        preds = []

        for b in results.boxes:

            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = b.conf[0].item()
            preds.append([x1, y1, x2, y2, conf])

        return preds

    def run_inference(self):

        cap = cv2.VideoCapture(int(self.source) if self.source == "0" else self.source)
        
        cv2.namedWindow("Ensemble Output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ensemble Output", self.disp_w, self.disp_h)
        
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret: break
            
            h, w = frame.shape[:2]
            scale = min(self.disp_w / w, self.disp_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Parallel model work
            res1 = self.model1(frame, verbose=False)[0]
            res2 = self.model2(frame, verbose=False)[0]
            res3 = self.model3(frame, verbose=False)[0]
            
            # Raw predictions
            preds1 = self._extract_boxes(res1)
            preds2 = self._extract_boxes(res2)
            preds3 = self._extract_boxes(res3)
            
            # Вычисляем максимальную уверенность для каждой модели на текущем кадре (или 0.0)
            conf1_max = max([p[4] for p in preds1] + [0.0])
            conf2_max = max([p[4] for p in preds2] + [0.0])
            conf3_max = max([p[4] for p in preds3] + [0.0])
            
            # Consensus aggregation
            final_boxes = self.ensemble.process(preds1, preds2, preds3)

            # Risk logic
            status = 0 # Safe

            if final_boxes:

                status = 2 # Fire

            elif preds1 or preds2 or preds3:

                status = 1 # Partial risk
            
            # Status visualization
            color_map = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}
            text_map = {0: "SAFE", 1: "RISK (NO CONSENSUS)", 2: "FIRE DETECTED"}

            cv2.putText(frame, f"STATUS: {text_map[status]}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_map[status], 4)
            cv2.putText(frame, f"Model 1 (Red): {conf1_max:.2f}", (20, new_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(frame, f"Model 2 (Blue): {conf2_max:.2f}", (20, new_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
            cv2.putText(frame, f"Model 3 (Yellow): {conf3_max:.2f}", (20, new_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            
            # Predict visualization
            for p in preds1:

                x1, y1, x2, y2 = map(int, p[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red for the 1st model
            
            for p in preds2:

                x1, y1, x2, y2 = map(int, p[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for the 2nd model
                
            for p in preds3:

                x1, y1, x2, y2 = map(int, p[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow for the 3rd model
                
            for b in final_boxes:

                x1, y1, x2, y2 = map(int, b[:4])
                conf = b[4]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4) # Consensus
                cv2.putText(frame, f"CONF: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            cv2.imshow("Ensemble Output", frame)

            if cv2.waitKey(1) & 0xFF == 27: # ESC to exit

                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    app = EnsembleFireSystem()
    app.root.mainloop()