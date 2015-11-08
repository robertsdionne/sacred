#!/usr/bin/env python3

import argparse
import json
import sys


def main():
  parser = argparse.ArgumentParser(description="Replace unknown words with UNK token.")
  parser.add_argument(
      '--vocabulary', default='data/large/GoogleNews-vectors-negative300/vocabulary.txt')
  arguments = parser.parse_args()

  with open(arguments.vocabulary) as vocabulary_file:
    vocabulary = set(vocabulary_file.read().splitlines())
    conversation = json.loads(sys.stdin.read())

    for turn in conversation['turn']:
      for utterance in turn['utterance']:
        words = map(lambda word: word if word in vocabulary else '(%s)' % word, utterance['text'].split())
        utterance['text'] = ' '.join(words)

    print(json.dumps(conversation))


if '__main__' == __name__:
  main()
