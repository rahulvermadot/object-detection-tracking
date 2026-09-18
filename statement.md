# Project Statement

## Problem Statement

Manually monitoring video feeds to identify and count objects, whether
people entering a space, vehicles passing a point, or items on a
conveyor, is slow, error-prone, and does not scale. Existing solutions
either rely on expensive proprietary surveillance systems or require
manually reviewing footage after the fact. There is a need for an
accessible, open pipeline that can detect objects of interest in real
time, keep track of each object individually as it moves through a
scene, and automatically tally counts without human intervention.

This project addresses that gap by building a real-time object
detection, tracking, and counting system using a pretrained deep
learning model (YOLO), a custom multi-object tracker, and a
configurable virtual line-crossing counter, all running on standard
consumer hardware without requiring a GPU or paid cloud service.

## Scope of the Project

**In scope:**
- Real-time object detection from a live webcam feed or a recorded
  video file, using a pretrained YOLO model restricted to a
  configurable set of object classes (e.g. person, car, cell phone,
  bottle).
- Multi-object tracking that assigns each detected object a persistent
  ID across frames, implemented from scratch using IoU overlap and
  centroid-distance matching.
- Counting objects as they cross a single configurable virtual line,
  with in/out totals broken down per object class.
- Annotated video output (bounding boxes, track IDs, the counting
  line, live totals) and a CSV summary log of final counts.
- Automated unit tests for the tracking and counting logic.

**Out of scope:**
- Object re-identification after an object fully leaves and re-enters
  the frame.
- Multiple simultaneous counting lines/zones or curved boundaries.
- Training a custom YOLO model from scratch (the project uses
  pretrained weights; fine-tuning is noted only as a future extension).
- Cloud deployment, multi-camera setups, or a web-based dashboard.
- Person re-identification or facial recognition of any kind.

## Target Users

- **Students and instructors** evaluating a computer vision project
  for coursework, wanting a system whose detection, tracking, and
  counting logic is fully explainable rather than dependent on an
  opaque third-party library.
- **Hobbyist developers** who want a working starting point for
  small-scale monitoring use cases, such as counting people entering a
  room or vehicles passing a driveway, without commercial surveillance
  software.
- **Small businesses or event organizers** interested in a low-cost,
  self-hosted way to get approximate footfall or object counts from a
  single camera, without sending video to a third-party cloud service.

## High-Level Features

- **Real-time detection** of multiple object classes using a
  pretrained YOLO model, running on CPU by default.
- **Persistent multi-object tracking** that keeps a stable ID on each
  object as it moves, even through brief occlusion, using a
  transparent IoU + centroid-distance matching algorithm.
- **Configurable virtual line counting** that tallies objects crossing
  a line in either direction, with a live per-class breakdown.
- **Annotated video output** showing bounding boxes, class labels,
  track IDs, the counting line, and running totals overlaid on the
  original footage.
- **CSV export** of final counts for easy reporting or further
  analysis.
- **Fully configurable pipeline** (detection confidence, tracked
  classes, tracker sensitivity, counting line position) via a single
  settings file, with no code changes required for common adjustments.
- **Automated test suite** validating tracking and counting behavior
  independent of any camera or live model inference.
