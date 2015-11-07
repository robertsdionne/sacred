#!/usr/bin/env python3

import argparse
import sys

from google.protobuf import json_format
from google.protobuf import text_format
from sacred.proto.conversation_pb2 import Conversation


BINARY = 'binary'
JSON = 'json'
TEXT = 'text'
FORMATS = [BINARY, JSON, TEXT]


def main():
  parser = argparse.ArgumentParser(description="Convert between proto3 formats.")
  parser.add_argument('--input_format', choices=FORMATS, default=TEXT, help="Input format.")
  parser.add_argument('--output_format', choices=FORMATS, default=JSON, help="Output format.")
  arguments = parser.parse_args()

  conversation = Conversation()

  if BINARY == arguments.input_format:
    conversation.ParseFromString(sys.stdin.read())
  elif JSON == arguments.input_format:
    json_format.Parse(sys.stdin.read(), conversation)
  else:
    text_format.Parse(sys.stdin.read(), conversation)

  if BINARY == arguments.output_format:
    sys.stdout.write(conversation.SerializeToString())
  elif JSON == arguments.output_format:
    print(json_format.MessageToJson(conversation))
  else:
    print(text_format.MessageToString(conversation, as_utf8=True))


if '__main__' == __name__:
  main()
