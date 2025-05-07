import grpc
import asyncio
import time
import sys
from collections import defaultdict
from concurrent import futures
import numpy as np

sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

class AggregatedServiceServicer(streaming_pb2_grpc.AggregatedServiceServicer):
    def __init__(self, num_workers=2):
        self.frame_queue = asyncio.Queue(maxsize=100)
        self.response_gateway = {}
        self.vehicle_results = {}
        self.plate_results = {}
        self.frame_images = {}
        self.num_workers = num_workers
        self.lock = asyncio.Lock()

    async def worker(self):
        while True:
            frame, response_queue = await self.frame_queue.get()
            frame_id = frame.frame_id
            print(f"🧠 Nhận frame {frame_id} từ gateway")
            
            async with self.lock:
                self.frame_images[frame_id] = frame.image_data
            self.frame_queue.task_done()

    async def ProcessVideo(self, request_iterator, context):
        asyncio.create_task(self.worker())

        async def receiver():
            print("🔄 Bắt đầu nhận frame từ Gateway...")
            async for frame in request_iterator:
                response_queue = asyncio.Queue()
                async with self.lock:
                    self.response_gateway[frame.frame_id] = response_queue
                await self.frame_queue.put((frame, response_queue))

        receiver_task = asyncio.create_task(receiver())
        try:
            while True:
                async with self.lock:
                    frame_ids = list(self.response_gateway.keys())

                for frame_id in frame_ids:
                    async with self.lock:
                        q = self.response_gateway.get(frame_id)
                    if q and not q.empty():
                        result = await q.get()
                        async with self.lock:
                            del self.response_gateway[frame_id]
                        yield result
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            print("🔚 Client ngắt kết nối.")
        finally:
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                print("🔚 Receiver task đã bị hủy.")

    async def ProcessVehicles(self, request_iterator, context):
        async for result in request_iterator:
            async with self.lock:
                print(f"🧠 Nhận kết quả từ VehicleServer cho frame {result.frame_id} lúc {time.time()}")
                self.vehicle_results[result.frame_id] = result
            yield streaming_pb2.Response(frame_id=result.frame_id, status="OK")
            await self.try_merge_and_send(result.frame_id)

    async def ProcessPlates(self, request_iterator, context):
        async for result in request_iterator:
            async with self.lock:
                print(f"🔳 Nhận kết quả từ PlateServer cho frame {result.frame_id} lúc {time.time()}")
                self.plate_results[result.frame_id] = result
            yield streaming_pb2.Response(frame_id=result.frame_id, status="OK")
            await self.try_merge_and_send(result.frame_id)

    async def try_merge_and_send(self, frame_id):
        async with self.lock:
            if (
                frame_id in self.vehicle_results
                and frame_id in self.plate_results
                and frame_id in self.response_gateway
            ):
                vehicle_result = self.vehicle_results.pop(frame_id)
                plate_result = self.plate_results.pop(frame_id)
                response_queue = self.response_gateway[frame_id]

        if 'vehicle_result' in locals() and 'plate_result' in locals():
            merged_result = await self.merge_results(frame_id, vehicle_result, plate_result)
            await response_queue.put(merged_result)
            print(f"✅ Đã gửi kết quả tổng hợp frame {frame_id}")

    async def merge_results(self, frame_id, vehicle_result, plate_result):
        print(f"🔄 Bắt đầu tổng hợp kết quả cho frame {frame_id}")
        merged_objects = []
        for track in vehicle_result.tracks:
            matched_plate = self.find_best_matching_plate(track.bbox, plate_result.plates)
            merged_objects.append(
                streaming_pb2.ObjectResult(
                    tracking_id=track.id,
                    vehicle_bbox=track.bbox,
                    plate_bbox=matched_plate.bbox if matched_plate else None,
                    plate_number=matched_plate.plate_number if matched_plate else "",
                    vehicle_class=track.class_id
                )
            )

        async with self.lock:
            image = self.frame_images.pop(frame_id, None)
        
        return streaming_pb2.AggregatedResult(
            frame_id=frame_id,
            image_data=image,
            objects=merged_objects
        )

    def find_best_matching_plate(self, vehicle_bbox, plates):
        def iou(box1, box2):
            xA = max(box1.x1, box2.x1)
            yA = max(box1.y1, box2.y1)
            xB = min(box1.x2, box2.x2)
            yB = min(box1.y2, box2.y2)
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
            boxBArea = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
            return interArea / (boxAArea + boxBArea - interArea + 1e-5)

        best = None
        best_score = 0.0
        for plate in plates:
            score = iou(vehicle_bbox, plate.bbox)
            if score > best_score:
                best_score = score
                best = plate
        return best if best_score > 0.01 else None

async def serve():
    server = grpc.aio.server()
    servicer = AggregatedServiceServicer(num_workers=4)
    streaming_pb2_grpc.add_AggregatedServiceServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:5005")
    await server.start()
    print("🚀 Main Server đang chạy trên cổng 5005...")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())

