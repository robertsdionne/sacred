#!/usr/bin/env bash

cat $1 | \
    ../data-science-at-the-command-line/tools/header -a 'time\tsender\trecipient\tutterance' | \
    csvjson --tabs | \
    bazel-bin/sacred/tools/group_by --key=sender | \
    jq -C '.[] | {time: .[0].time, speaker: .[0].sender, utterance: [.[].utterance]}'
