import grpc
import asyncio
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import sys
import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ['FLAGS_logging_level'] = '3'
sys.path.append("generated")
import paddle
from generated import streaming_pb2, streaming_pb2_grpc

device = paddle.device.get_device()
print(f"🔹 Đang sử dụng: {device}")

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
            if score > 0.75:
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
    def __init__(self, num_workers=2):
        self.ocr_model = PaddleOCR(use_angle_cls=True, use_gpu=True, 
                                   lang='en', enable_mkldnn=False, show_log=False)
        self.warmup_model()
        
        self.queue = asyncio.Queue()
        self.send_queue = asyncio.Queue()
        
        self.main_channel = grpc.aio.insecure_channel("localhost:5005")
        self.main_stub = streaming_pb2_grpc.AggregatedServiceStub(self.main_channel)
        
        self.num_workers = num_workers
        self.response_queue = {}
        self.workers_started = False
        
    def warmup_model(self):
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        print("🔥 Warmup OCR model...")
        _ = self.ocr_model.ocr(dummy_image_rgb)
        print("✅ Warmup hoàn tất.")    
    
    async def stream_sender(self):
        async def request_generator():
            while True:
                result = await self.send_queue.get()
                try:
                    send_time = time.time()
                    print(f"📤 Gửi frame ID {result.frame_id} lúc {send_time:.2f}")
                    yield result
                except Exception as e:
                    print(f"❌ Lỗi khi gửi frame ID {result.frame_id}: {e}")
                finally:
                    self.send_queue.task_done()
                    
        async def response_handler(call):
            try:
                async for response in call:
                    # Nếu bạn muốn xử lý kết quả từ AggregatedService
                    pass
            except grpc.aio.AioRpcError as e:
                print(f"❌ Lỗi phản hồi từ AggregatedService: {e.code()} - {e.details()}")
                
        while True:
            try:
                print("🔁 Kết nối tới AggregatedService...")
                call = self.main_stub.ProcessPlates(request_generator())
                await response_handler(call)
            except grpc.aio.AioRpcError as e:
                print(f"❌ Mất kết nối tới AggregatedService: {e.code()} - {e.details()}")
            except Exception as e:
                print(f"❌ Lỗi không xác định khi gửi stream: {e}")
            print("🕒 Đợi 3 giây trước khi thử kết nối lại...")
            await asyncio.sleep(3)
        
    def start_workers(self):
        if not self.workers_started:
            for i in range(self.num_workers):
                asyncio.create_task(self.worker(i))
            self.workers_started = True
            
        asyncio.create_task(self.stream_sender())
    
    async def RecognizePlate(self, request_iterator, context):
        self.start_workers()
        
        async def receiver():
            print("🔄 Bắt đầu nhận frame từ PlateService...")
            async for plateDetection in request_iterator:
                response_queue = asyncio.Queue()
                self.response_queue[plateDetection.frame_id] = response_queue
                await self.queue.put((plateDetection, response_queue))
        
        receiver_task = asyncio.create_task(receiver())
        try:
            while True:
                for frame_id in list(self.response_queue.keys()):
                    q = self.response_queue[frame_id]
                    if not q.empty():
                        result = await q.get()
                        del self.response_queue[frame_id]
                        yield streaming_pb2.Response(frame_id=frame_id, status="OK")
                await asyncio.sleep(0.01)  # Giảm tải CPU
        except asyncio.CancelledError:
            print("🔚 Client ngắt kết nối.")
        finally:
            receiver_task.cancel()
            
    async def worker(self, worker_id):
        while True:
            try:
                plateDetection, response_queue = await self.queue.get()
                frame_id = plateDetection.frame_id
                
                print(f"Thời điểm worker-{worker_id} nhận frame {frame_id}:", time.time())
            
                plates = plateDetection.plates
                plates_recognition = []
                
                for plate in plates:
                    try:
                        cropped_plate = np.frombuffer(plate.cropped_plate, np.uint8)
                        cropped_plate = cv2.imdecode(cropped_plate, cv2.IMREAD_COLOR)
                        resized_plate_crop = cv2.resize(cropped_plate, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                        if resized_plate_crop is None:
                            raise ValueError("Không thể giải mã ảnh biển số.")
                    except Exception as e:
                        print(f"❌ Lỗi khi giải mã ảnh biển số trong frame {frame_id}: {e}")
                        self.queue.task_done()
                        continue
                    
                    start_time = time.time()
                    text, score = await asyncio.to_thread(read_license_plate, self.ocr_model, resized_plate_crop)
                    print(f"{text} - {score}")
                    inference_time = time.time() - start_time
                    print(f"⏱️ Thời gian OCR xử lý: {inference_time:.4f} giây")
                    
                    plates_recognition.append(streaming_pb2.PlateRecognition(
                        bbox=plate.bbox,
                        plate_number=text,
                        cropped_plate=plate.cropped_plate
                    ))
                
                # print(plates_recognition)
                ocr_result = streaming_pb2.OCRResult(
                    frame_id=frame_id,
                    plates=plates_recognition
                )
                
                await response_queue.put(ocr_result)
                await self.send_queue.put(ocr_result)
                self.queue.task_done()
                print(f"✅ Worker-{worker_id} đã xử lý frame {frame_id} và gửi kết quả.")
                print(f"⏱️ Thời điểm worker-{worker_id} hoàn tất: {time.time()}")
            except asyncio.QueueEmpty:
                print(f"⚠️ Worker-{worker_id} không có frame nào để xử lý.")
            except Exception as e:
                print(f"❌ Lỗi trong worker-{worker_id}: {e}")
                await asyncio.sleep(1)

async def serve():
    server = grpc.aio.server()
    servicer = OCRServicer(num_workers=16)
    streaming_pb2_grpc.add_OCRServiceServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:5003")          
    await server.start()
    print("🚀 OCR Server đang chạy trên cổng 5003...")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
