# GDELT dual-track historical plan

## Track A — production historical lockbox

The production historical lockbox uses the official GDELT 1.0 daily event stream at `events/{YYYYMMDD}.export.CSV.zip` while GDELT 2.0 is independently acquired and validated.

GDELT 1.0 daily files are named for the previous day's event-discovery date and posted the following morning. The pipeline therefore never treats `SQLDATE` or the 8-digit file date as availability. It assigns a conservative availability boundary of **12:00 UTC on the day after the file date**. Original article publication time remains unknown.

The production source ID is `GDELT_1_Daily_Event_Database`.

## Track B — independent GDELT 2.0 acquisition

Track B is acquired from the official GDELT 2.0 master file lists. GDELT 2.0 is a 15-minute event stream and its `DATEADDED` field is retained as the information-availability timestamp. English and translingual streams are kept distinguishable.

Track B is research staging only. Successful downloads do not authorize substitution into the production lockbox. Before promotion, schema, coverage, source identity, availability semantics, deduplication and point-in-time behavior must pass explicit tests.

## Rule

A and B can be downloaded in parallel, but only Track A is used for the current production report. Track B must pass its own validation gate before it can replace or augment A.
