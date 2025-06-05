# ws_server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Query
from pymongo import MongoClient
import os
import uvicorn

app = FastAPI()
# Kết nối tới MongoDB
try:
    client = MongoClient("mongodb+srv://***:*****@cluster0.89utstf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["vehicle_tracking_db"]
    collection = db["detection_logs"]
    print("✅ Kết nối tới MongoDB thành công")
except Exception as e:
    print(f"❌ Kết nối tới MongoDB thất bại: {e}")

# Cho phép frontend truy cập từ trình duyệt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Danh sách kết nối
gateway_clients = []       # WebSocket từ gateway_server
frontend_clients = []      # WebSocket từ các client trình duyệt

@app.websocket("/ws/gateway")
async def websocket_gateway(websocket: WebSocket):
    await websocket.accept()
    gateway_clients.append(websocket)
    print("🚀 Gateway connected")

    try:
        while True:
            data = await websocket.receive_text()
            print("📨 Dữ liệu từ gateway received, gửi tới frontend clients...")
            to_remove = []
            for client in frontend_clients:
                try:
                    await client.send_text(data)
                except Exception:
                    to_remove.append(client)
            for client in to_remove:
                frontend_clients.remove(client)
    except WebSocketDisconnect:
        gateway_clients.remove(websocket)
        print("❌ Gateway disconnected")


@app.websocket("/ws/client")
async def websocket_client(websocket: WebSocket):
    await websocket.accept()
    frontend_clients.append(websocket)
    print(f"🟢 Frontend client connected. Total: {len(frontend_clients)}")
    try:
        while True:
            await websocket.receive_text()  # giữ kết nối
    except WebSocketDisconnect:
        frontend_clients.remove(websocket)
        print(f"🔴 Frontend client disconnected. Total: {len(frontend_clients)}")

@app.get("/frame/{frame_name}")
async def get_frame(frame_name: str):
    file_path = f"/tmp/saved_frames/{frame_name}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    else:
        print(f"❌ Không tìm thấy file {file_path}")
        return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/search_plate")
async def search_plate(plate: str = Query(...)):
    results = []
    query = {"objects.plate_number": plate.upper()}  # hoặc plate.lower() nếu bạn lưu thường
    cursor = collection.find(query).sort("timestamp", -1).limit(50)

    for doc in cursor:
        for obj in doc["objects"]:
            if obj.get("plate_number", "").upper() == plate.upper():
                results.append({
                    "frame_id": doc["frame_id"],
                    "image_link": f"http://localhost:8000/frame/{doc['image_link']}",
                    "timestamp": doc["timestamp"],
                    "tracking_id": obj["tracking_id"],
                    "vehicle_class": obj["vehicle_class"],
                    "plate_number": obj["plate_number"],
                    "vehicle_box": obj["vehicle_bbox"],
                    "plate_box": obj["plate_bbox"]
                })
    print(f"🔍 Tìm thấy {len(results)} kết quả cho biển số {plate.upper()}")
    return JSONResponse(content=results)

if __name__ == "__main__":
    uvicorn.run("ws_server:app", host="0.0.0.0", port=8000, reload=True)
