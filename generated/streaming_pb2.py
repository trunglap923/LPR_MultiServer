# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: streaming.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fstreaming.proto\x12\tstreaming\"6\n\x04\x42\x42ox\x12\n\n\x02x1\x18\x01 \x01(\x02\x12\n\n\x02y1\x18\x02 \x01(\x02\x12\n\n\x02x2\x18\x03 \x01(\x02\x12\n\n\x02y2\x18\x04 \x01(\x02\"@\n\x05\x46rame\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x03\x12\x12\n\nimage_data\x18\x02 \x01(\x0c\x12\x11\n\ttimestamp\x18\x03 \x01(\x03\",\n\x08Response\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x03\x12\x0e\n\x06status\x18\x02 \x01(\t\"b\n\x0f\x44\x65tectedVehicle\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x1d\n\x04\x62\x62ox\x18\x02 \x01(\x0b\x32\x0f.streaming.BBox\x12\x10\n\x08\x63lass_id\x18\x03 \x01(\x05\x12\x12\n\nconfidence\x18\x04 \x01(\x02\"f\n\x10VehicleDetection\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x03\x12\x12\n\nimage_data\x18\x02 \x01(\x0c\x12,\n\x08vehicles\x18\x03 \x03(\x0b\x32\x1a.streaming.DetectedVehicle\"k\n\rDetectedPlate\x12\x1d\n\x04\x62\x62ox\x18\x01 \x01(\x0b\x32\x0f.streaming.BBox\x12\x10\n\x08\x63lass_id\x18\x02 \x01(\x05\x12\x12\n\nconfidence\x18\x03 \x01(\x02\x12\x15\n\rcropped_plate\x18\x04 \x01(\x0c\"`\n\x0ePlateDetection\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x03\x12\x12\n\nimage_data\x18\x02 \x01(\x0c\x12(\n\x06plates\x18\x03 \x03(\x0b\x32\x18.streaming.DetectedPlate\"[\n\rTrackedObject\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x1d\n\x04\x62\x62ox\x18\x02 \x01(\x0b\x32\x0f.streaming.BBox\x12\x10\n\x08\x63lass_id\x18\x03 \x01(\x05\x12\r\n\x05score\x18\x04 \x01(\x02\"L\n\x0eTrackingResult\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x03\x12(\n\x06tracks\x18\x02 \x03(\x0b\x32\x18.streaming.TrackedObject\"^\n\x10PlateRecognition\x12\x1d\n\x04\x62\x62ox\x18\x01 \x01(\x0b\x32\x0f.streaming.BBox\x12\x14\n\x0cplate_number\x18\x02 \x01(\t\x12\x15\n\rcropped_plate\x18\x03 \x01(\x0c\"J\n\tOCRResult\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x03\x12+\n\x06plates\x18\x02 \x03(\x0b\x32\x1b.streaming.PlateRecognition\"\x9c\x01\n\x0cObjectResult\x12\x13\n\x0btracking_id\x18\x01 \x01(\x05\x12%\n\x0cvehicle_bbox\x18\x02 \x01(\x0b\x32\x0f.streaming.BBox\x12#\n\nplate_bbox\x18\x03 \x01(\x0b\x32\x0f.streaming.BBox\x12\x14\n\x0cplate_number\x18\x04 \x01(\t\x12\x15\n\rvehicle_class\x18\x05 \x01(\x05\"u\n\x10\x41ggregatedResult\x12\x10\n\x08\x66rame_id\x18\x01 \x01(\x03\x12\x12\n\nimage_data\x18\x02 \x01(\x0c\x12\x11\n\ttimestamp\x18\x03 \x01(\x03\x12(\n\x07objects\x18\x04 \x03(\x0b\x32\x17.streaming.ObjectResult2V\n\x17VehicleDetectionService\x12;\n\x0e\x44\x65tectVehicles\x12\x10.streaming.Frame\x1a\x13.streaming.Response(\x01\x30\x01\x32R\n\x15PlateDetectionService\x12\x39\n\x0c\x44\x65tectPlates\x12\x10.streaming.Frame\x1a\x13.streaming.Response(\x01\x30\x01\x32W\n\x0fTrackingService\x12\x44\n\x0cTrackObjects\x12\x1b.streaming.VehicleDetection\x1a\x13.streaming.Response(\x01\x30\x01\x32R\n\nOCRService\x12\x44\n\x0eRecognizePlate\x12\x19.streaming.PlateDetection\x1a\x13.streaming.Response(\x01\x30\x01\x32O\n\x15VideoStreamingService\x12\x36\n\x0bStreamVideo\x12\x10.streaming.Frame\x1a\x13.streaming.Response(\x01\x32\xdd\x01\n\x11\x41ggregatedService\x12\x41\n\x0cProcessVideo\x12\x10.streaming.Frame\x1a\x1b.streaming.AggregatedResult(\x01\x30\x01\x12>\n\rProcessPlates\x12\x14.streaming.OCRResult\x1a\x13.streaming.Response(\x01\x30\x01\x12\x45\n\x0fProcessVehicles\x12\x19.streaming.TrackingResult\x1a\x13.streaming.Response(\x01\x30\x01\x62\x06proto3')



_BBOX = DESCRIPTOR.message_types_by_name['BBox']
_FRAME = DESCRIPTOR.message_types_by_name['Frame']
_RESPONSE = DESCRIPTOR.message_types_by_name['Response']
_DETECTEDVEHICLE = DESCRIPTOR.message_types_by_name['DetectedVehicle']
_VEHICLEDETECTION = DESCRIPTOR.message_types_by_name['VehicleDetection']
_DETECTEDPLATE = DESCRIPTOR.message_types_by_name['DetectedPlate']
_PLATEDETECTION = DESCRIPTOR.message_types_by_name['PlateDetection']
_TRACKEDOBJECT = DESCRIPTOR.message_types_by_name['TrackedObject']
_TRACKINGRESULT = DESCRIPTOR.message_types_by_name['TrackingResult']
_PLATERECOGNITION = DESCRIPTOR.message_types_by_name['PlateRecognition']
_OCRRESULT = DESCRIPTOR.message_types_by_name['OCRResult']
_OBJECTRESULT = DESCRIPTOR.message_types_by_name['ObjectResult']
_AGGREGATEDRESULT = DESCRIPTOR.message_types_by_name['AggregatedResult']
BBox = _reflection.GeneratedProtocolMessageType('BBox', (_message.Message,), {
  'DESCRIPTOR' : _BBOX,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.BBox)
  })
_sym_db.RegisterMessage(BBox)

Frame = _reflection.GeneratedProtocolMessageType('Frame', (_message.Message,), {
  'DESCRIPTOR' : _FRAME,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.Frame)
  })
_sym_db.RegisterMessage(Frame)

Response = _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSE,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.Response)
  })
_sym_db.RegisterMessage(Response)

DetectedVehicle = _reflection.GeneratedProtocolMessageType('DetectedVehicle', (_message.Message,), {
  'DESCRIPTOR' : _DETECTEDVEHICLE,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.DetectedVehicle)
  })
_sym_db.RegisterMessage(DetectedVehicle)

VehicleDetection = _reflection.GeneratedProtocolMessageType('VehicleDetection', (_message.Message,), {
  'DESCRIPTOR' : _VEHICLEDETECTION,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.VehicleDetection)
  })
_sym_db.RegisterMessage(VehicleDetection)

DetectedPlate = _reflection.GeneratedProtocolMessageType('DetectedPlate', (_message.Message,), {
  'DESCRIPTOR' : _DETECTEDPLATE,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.DetectedPlate)
  })
_sym_db.RegisterMessage(DetectedPlate)

PlateDetection = _reflection.GeneratedProtocolMessageType('PlateDetection', (_message.Message,), {
  'DESCRIPTOR' : _PLATEDETECTION,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.PlateDetection)
  })
_sym_db.RegisterMessage(PlateDetection)

TrackedObject = _reflection.GeneratedProtocolMessageType('TrackedObject', (_message.Message,), {
  'DESCRIPTOR' : _TRACKEDOBJECT,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.TrackedObject)
  })
_sym_db.RegisterMessage(TrackedObject)

TrackingResult = _reflection.GeneratedProtocolMessageType('TrackingResult', (_message.Message,), {
  'DESCRIPTOR' : _TRACKINGRESULT,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.TrackingResult)
  })
_sym_db.RegisterMessage(TrackingResult)

PlateRecognition = _reflection.GeneratedProtocolMessageType('PlateRecognition', (_message.Message,), {
  'DESCRIPTOR' : _PLATERECOGNITION,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.PlateRecognition)
  })
_sym_db.RegisterMessage(PlateRecognition)

OCRResult = _reflection.GeneratedProtocolMessageType('OCRResult', (_message.Message,), {
  'DESCRIPTOR' : _OCRRESULT,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.OCRResult)
  })
_sym_db.RegisterMessage(OCRResult)

ObjectResult = _reflection.GeneratedProtocolMessageType('ObjectResult', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTRESULT,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.ObjectResult)
  })
_sym_db.RegisterMessage(ObjectResult)

AggregatedResult = _reflection.GeneratedProtocolMessageType('AggregatedResult', (_message.Message,), {
  'DESCRIPTOR' : _AGGREGATEDRESULT,
  '__module__' : 'streaming_pb2'
  # @@protoc_insertion_point(class_scope:streaming.AggregatedResult)
  })
_sym_db.RegisterMessage(AggregatedResult)

_VEHICLEDETECTIONSERVICE = DESCRIPTOR.services_by_name['VehicleDetectionService']
_PLATEDETECTIONSERVICE = DESCRIPTOR.services_by_name['PlateDetectionService']
_TRACKINGSERVICE = DESCRIPTOR.services_by_name['TrackingService']
_OCRSERVICE = DESCRIPTOR.services_by_name['OCRService']
_VIDEOSTREAMINGSERVICE = DESCRIPTOR.services_by_name['VideoStreamingService']
_AGGREGATEDSERVICE = DESCRIPTOR.services_by_name['AggregatedService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _BBOX._serialized_start=30
  _BBOX._serialized_end=84
  _FRAME._serialized_start=86
  _FRAME._serialized_end=150
  _RESPONSE._serialized_start=152
  _RESPONSE._serialized_end=196
  _DETECTEDVEHICLE._serialized_start=198
  _DETECTEDVEHICLE._serialized_end=296
  _VEHICLEDETECTION._serialized_start=298
  _VEHICLEDETECTION._serialized_end=400
  _DETECTEDPLATE._serialized_start=402
  _DETECTEDPLATE._serialized_end=509
  _PLATEDETECTION._serialized_start=511
  _PLATEDETECTION._serialized_end=607
  _TRACKEDOBJECT._serialized_start=609
  _TRACKEDOBJECT._serialized_end=700
  _TRACKINGRESULT._serialized_start=702
  _TRACKINGRESULT._serialized_end=778
  _PLATERECOGNITION._serialized_start=780
  _PLATERECOGNITION._serialized_end=874
  _OCRRESULT._serialized_start=876
  _OCRRESULT._serialized_end=950
  _OBJECTRESULT._serialized_start=953
  _OBJECTRESULT._serialized_end=1109
  _AGGREGATEDRESULT._serialized_start=1111
  _AGGREGATEDRESULT._serialized_end=1228
  _VEHICLEDETECTIONSERVICE._serialized_start=1230
  _VEHICLEDETECTIONSERVICE._serialized_end=1316
  _PLATEDETECTIONSERVICE._serialized_start=1318
  _PLATEDETECTIONSERVICE._serialized_end=1400
  _TRACKINGSERVICE._serialized_start=1402
  _TRACKINGSERVICE._serialized_end=1489
  _OCRSERVICE._serialized_start=1491
  _OCRSERVICE._serialized_end=1573
  _VIDEOSTREAMINGSERVICE._serialized_start=1575
  _VIDEOSTREAMINGSERVICE._serialized_end=1654
  _AGGREGATEDSERVICE._serialized_start=1657
  _AGGREGATEDSERVICE._serialized_end=1878
# @@protoc_insertion_point(module_scope)
