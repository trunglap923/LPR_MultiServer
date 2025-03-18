import grpc
from concurrent import futures
import time 
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import sys
import numpy as np
sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Đang sử dụng: {device}")

class TrackingServicer(streaming_pb2_grpc.TrackingServiceServicer):
    def __init__(self):
        self.tracker = DeepSort(max_age=20, embedder_gpu=device == "cuda")
        self.frame_count = 0  # Biến đếm số lượng frame đã nhận
        self.last_frame_time = time.time()  # Thời gian nhận frame cuối cùng
    
    def TrackObjects(self, request_iterator, context):
        for vehicleDetection in request_iterator:
            # Tăng bộ đếm frame
            self.frame_count += 1
            # Kiểm tra thời gian từ frame cuối cùng
            current_time = time.time()
            if current_time - self.last_frame_time > 10:  # Nếu quá 10 giây
                print("⚠️ Không nhận được frame trong 10 giây, reset frame_count về 0.")
                self.frame_count = 1  # Reset frame_count
            
            # Cập nhật thời gian nhận frame cuối cùng
            self.last_frame_time = current_time
            
            frame_id = vehicleDetection.frame_id
            image = np.frombuffer(vehicleDetection.image_data, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            vehicles = vehicleDetection.vehicles
            
            formatted_vehicles = [
                [[v.bbox[0], v.bbox[1], v.bbox[2] - v.bbox[0], v.bbox[3] - v.bbox[1]], v.confidence, v.class_id]
                for v in vehicles
            ]
            
            start_time = time.time()
            with torch.no_grad():
                tracks = self.tracker.update_tracks(formatted_vehicles, frame=image)
            inference_time = time.time() - start_time
            print(f"⏱️ Thời gian Tracking phương tiện: {inference_time:.4f} giây")

            tracked_objects = [
                streaming_pb2.TrackedObject(
                    id=int(track.track_id),
                    bbox=list(track.to_ltrb()),
                    class_id=track.get_det_class(),
                    score=track.get_det_conf() or 0
                )
                for track in tracks if track.is_confirmed()
            ]
            
            # Hiển thị frame_id, số lượng track và tổng số frame đã nhận
            print(f"Frame ID: {frame_id}")
            print(f"Số lượng track: {len(tracked_objects)}")
            print(f"Tổng số frame đã nhận: {self.frame_count}")
            
            tracking_result = streaming_pb2.TrackingResult(
                frame_id=frame_id,
                tracks=tracked_objects,
                success=True
            )
            yield tracking_result
                
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    streaming_pb2_grpc.add_TrackingServiceServicer_to_server(TrackingServicer(), server)
    server.add_insecure_port("[::]:5004")          
    server.start()
    print("🚀 Tracking Server đang chạy trên cổng 5004...")
    server.wait_for_termination()
    
if __name__ == "__main__":
    serve()