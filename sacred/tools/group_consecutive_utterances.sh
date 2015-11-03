#!/usr/bin/env bash

cat data/large/ubuntu_csvfiles/dialogs/188/1.tsv | \
    ../data-science-at-the-command-line/tools/header -a 'time\tsender\trecipient\tutterance' | \
    csvjson --tabs | \
    bazel-bin/sacred/tools/group_consecutive_utterances | \
    jq '.[] | {speaker: .[].sender, utterance: [.[].utterance]}'
