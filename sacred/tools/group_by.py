#!/usr/bin/env python3

import argparse
import itertools
import json
import operator
import sys


def main():
  parser = argparse.ArgumentParser(description='Group JSON by key.')
  parser.add_argument('--key', help='The key.')
  arguments = parser.parse_args()

  records = json.loads(sys.stdin.read())
  grouped = itertools.groupby(records, lambda record: record[arguments.key])
  print json.dumps([list(group) for key, group in grouped])


if '__main__' == __name__:
  main()
