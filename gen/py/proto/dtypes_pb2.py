# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/dtypes.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12proto/dtypes.proto\x12\nmlzoomcamp\"\x1a\n\tBytesList\x12\r\n\x05value\x18\x01 \x03(\x0c\"\x1e\n\tFloatList\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\"\x1e\n\tInt64List\x12\x11\n\x05value\x18\x01 \x03(\x03\x42\x02\x10\x01\x62\x06proto3')



_BYTESLIST = DESCRIPTOR.message_types_by_name['BytesList']
_FLOATLIST = DESCRIPTOR.message_types_by_name['FloatList']
_INT64LIST = DESCRIPTOR.message_types_by_name['Int64List']
BytesList = _reflection.GeneratedProtocolMessageType('BytesList', (_message.Message,), {
  'DESCRIPTOR' : _BYTESLIST,
  '__module__' : 'proto.dtypes_pb2'
  # @@protoc_insertion_point(class_scope:mlzoomcamp.BytesList)
  })
_sym_db.RegisterMessage(BytesList)

FloatList = _reflection.GeneratedProtocolMessageType('FloatList', (_message.Message,), {
  'DESCRIPTOR' : _FLOATLIST,
  '__module__' : 'proto.dtypes_pb2'
  # @@protoc_insertion_point(class_scope:mlzoomcamp.FloatList)
  })
_sym_db.RegisterMessage(FloatList)

Int64List = _reflection.GeneratedProtocolMessageType('Int64List', (_message.Message,), {
  'DESCRIPTOR' : _INT64LIST,
  '__module__' : 'proto.dtypes_pb2'
  # @@protoc_insertion_point(class_scope:mlzoomcamp.Int64List)
  })
_sym_db.RegisterMessage(Int64List)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _FLOATLIST.fields_by_name['value']._options = None
  _FLOATLIST.fields_by_name['value']._serialized_options = b'\020\001'
  _INT64LIST.fields_by_name['value']._options = None
  _INT64LIST.fields_by_name['value']._serialized_options = b'\020\001'
  _BYTESLIST._serialized_start=34
  _BYTESLIST._serialized_end=60
  _FLOATLIST._serialized_start=62
  _FLOATLIST._serialized_end=92
  _INT64LIST._serialized_start=94
  _INT64LIST._serialized_end=124
# @@protoc_insertion_point(module_scope)