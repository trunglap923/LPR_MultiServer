import grpc
from concurrent import futures
import time
import cv2
import numpy as np
import torch
import sys
sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

class VideoStreamingServicer(streaming_pb2_grpc.VideoStreamingServiceServicer):
    def __init__(self):
        self.vehicle_channel = grpc.insecure_channel("localhost:5001")
        self.vehicle_stub = streaming_pb2_grpc.VehicleDetectionServiceStub(self.vehicle_channel)
        self.plate_channel = grpc.insecure_channel("localhost:5002")
        self.plate_stub = streaming_pb2_grpc.PlateDetectionServiceStub(self.plate_channel)
        self.frame_count = 0  # Biến đếm số lượng frame đã nhận
        self.last_frame_time = time.time()  # Thời gian nhận frame cuối cùng
    
    def StreamVideo(self, request_iterator, context):
        for frame in request_iterator:
            # Tăng bộ đếm frame
            self.frame_count += 1
            # Kiểm tra thời gian từ frame cuối cùng
            current_time = time.time()
            if current_time - self.last_frame_time > 10:  # Nếu quá 10 giây
                print("⚠️ Không nhận được frame trong 10 giây, reset frame_count về 0.")
                self.frame_count = 1  # Reset frame_count
            
            # Cập nhật thời gian nhận frame cuối cùng
            self.last_frame_time = current_time

            

            print(f"Server: Received Frame ID: {frame.frame_id}")  # Ghi log frame_id
            vehicle_responses = self.vehicle_stub.DetectVehicles(iter([frame]))
            plate_responses = self.plate_stub.DetectPlates(iter([frame]))
            
            for vehicle_response in vehicle_responses:
                num_vehicles = len(vehicle_response.vehicles)
                print(f"Frame ID: {vehicle_response.frame_id}")
                print(f"  - Số lượng phương tiện phát hiện: {num_vehicles}")
                
            for plate_response in plate_responses:
                num_plates = len(plate_response.plates)
                print(f"Frame ID: {plate_response.frame_id}")
                print(f"  - Số lượng phương tiện phát hiện: {num_plates}")
            
            # Hiển thị tổng số frame đã nhận
            print(f"Tổng số frame đã nhận: {self.frame_count}")
            
            yield frame

    # def StreamVideo(self, request_iterator, context):
    #     for frame in request_iterator:
    #         # Kiểm tra thời gian từ frame cuối cùng
    #         current_time = time.time()
    #         if current_time - self.last_frame_time > 10:  # Nếu quá 10 giây
    #             print("⚠️ Không nhận được frame trong 10 giây, reset frame_count về 0.")
    #             self.frame_count = 0  # Reset frame_count
    #         
    #         # Cập nhật thời gian nhận frame cuối cùng
    #         self.last_frame_time = current_time
    #         
    #         # Tăng bộ đếm frame
    #         self.frame_count += 1
    #         
    #         print(f"Server: Received Frame ID: {frame.frame_id}")  # Ghi log frame_id
    #         vehicle_responses = self.vehicle_stub.DetectVehicles(iter([frame]))
    #         plate_responses = self.plate_stub.DetectPlates(iter([frame]))
    #         
    #         for vehicle_response, plate_response in zip(vehicle_responses, plate_responses):
    #             num_vehicles = len(vehicle_response.vehicles)
    #             num_plates = len(plate_response.plates)
    #             print(f"Frame ID: {vehicle_response.frame_id}")
    #             print(f"  - Số lượng phương tiện phát hiện: {num_vehicles}")
    #             print(f"  - Số lượng biển số phát hiện: {num_plates}")
    #         
    #         # Hiển thị tổng số frame đã nhận
    #         print(f"Tổng số frame đã nhận: {self.frame_count}")
    #         
    #         yield frame
            
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    streaming_pb2_grpc.add_VideoStreamingServiceServicer_to_server(VideoStreamingServicer(), server)
    server.add_insecure_port("[::]:5000")
    server.start()
    print("🚀 gRPC Video Streaming Server đang chạy trên cổng 5000...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
