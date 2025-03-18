import grpc
from concurrent import futures
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append("generated")
import paddle
from generated import streaming_pb2, streaming_pb2_grpc

print(paddle.device.get_device())

def read_license_plate(ocr, license_plate):
    result = ocr.ocr(license_plate)
    license_plate_text = ''
    license_plate_score = 1
    if result[0] is not None:
        texts_use = []
        for r in result[0]:
            score = r[1][1]

            if np.isnan(score):
                score = 0
            if score > 0.6:
                license_plate_score *= score
                pattern = re.compile('[\W]')
                text = pattern.sub('', r[1][0])
                text = text.replace("???", "")
                text = text.replace("O", "0")
                text = text.replace("粤", "")
                texts_use.append(text)

        texts_sorted = sorted(texts_use, key=len)
        license_plate_text = ''.join(texts_sorted).strip()

        return license_plate_text, license_plate_score
    return '', 0

class OCRServicer(streaming_pb2_grpc.OCRServiceServicer):
    def __init__(self):
        self.ocr_model = PaddleOCR(use_angle_cls=True, use_gpu=True, 
                                   lang='en', enable_mkldnn=False)
        self.frame_count = 0  # Biến đếm số lượng frame đã nhận
        self.last_frame_time = time.time()  # Thời gian nhận frame cuối cùng
    
    def RecognizePlate(self, request_iterator, context):
        for plateDetection in request_iterator:
            # Tăng bộ đếm frame
            self.frame_count += 1
            # Kiểm tra thời gian từ frame cuối cùng
            current_time = time.time()
            if current_time - self.last_frame_time > 10:  # Nếu quá 10 giây
                print("⚠️ Không nhận được frame trong 10 giây, reset frame_count về 0.")
                self.frame_count = 1  # Reset frame_count
            
            # Cập nhật thời gian nhận frame cuối cùng
            self.last_frame_time = current_time
            

            frame_id = plateDetection.frame_id
            print(f"Frame {frame_id}")
            plates = plateDetection.plates
            plates_recognition = []
            for plate in plates:
                cropped_plate = np.frombuffer(plate.cropped_plate, np.uint8)
                cropped_plate = cv2.imdecode(cropped_plate, cv2.IMREAD_COLOR)
                resized_plate_crop = cv2.resize(cropped_plate, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                
                start_time = time.time()
                text, score = read_license_plate(self.ocr_model, resized_plate_crop)
                print(f"{text} - {score}")
                inference_time = time.time() - start_time
                print(f"⏱️ Thời gian OCR xử lý: {inference_time:.4f} giây")
                
                plates_recognition.append(streaming_pb2.PlateRecognition(
                    bbox=plate.bbox,
                    plate_number=text,
                    cropped_plate=plate.cropped_plate
                ))
            
            ocr_result = streaming_pb2.OCRResult(
                frame_id=frame_id,
                plates=plates_recognition,
                success=True
            )
            
            # Hiển thị frame_id và tổng số frame đã nhận
            print(f"Frame ID: {frame_id}")
            print(f"Tổng số frame đã nhận: {self.frame_count}")
            
            yield ocr_result

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    streaming_pb2_grpc.add_OCRServiceServicer_to_server(OCRServicer(), server)
    server.add_insecure_port("[::]:5003")          
    server.start()
    print("🚀 OCR Server đang chạy trên cổng 5003...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()