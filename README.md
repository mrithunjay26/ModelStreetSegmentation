# Street Scene Segmentation Demo

TensorFlow.js semantic segmentation demo that identifies objects and regions in street-scene images directly in the browser. The model produces a 128 x 128 class map across 31 urban-scene labels including roads, buildings, cars, sky, trees, pedestrians, traffic lights, and bicyclists.

This project shows how a computer-vision model can be exported into a static web app and used for client-side inference without a Python backend.

## What It Shows

- TensorFlow.js graph model running in the browser
- Image upload and local inference through static HTML/JavaScript
- 31-class semantic segmentation output
- Canvas-based visualization of predicted class IDs
- Efficient model hosting through local shard files

## Model Summary

| Item | Detail |
| --- | --- |
| Runtime | TensorFlow.js |
| Model format | `graph-model` |
| Input | 256 x 256 RGB image |
| Output | 128 x 128 x 31 segmentation tensor |
| Classes | 31 street-scene classes |
| Weight shards | 5 local `.bin` files |

Example classes include:

```text
Road, Sidewalk, TrafficLight, Bicyclist, Car, Pedestrian,
Building, Tree, Sky, RoadShoulder, SignSymbol, TrafficCone
```

## Run Locally

Serve the repository over HTTP:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

Upload a 256 x 256 street-scene image or use the bundled sample image, then click `predict`. The page draws the predicted segmentation map onto the canvas.

## Repository Layout

```text
index.html              Browser UI, inference code, and canvas rendering
model.json              TensorFlow.js graph model manifest
classes.json            31-class label mapping
group1-shard*.bin       Model weight shards
image.jpg               Sample street-scene input
```

## Verification

Current automated checks performed:

- Parsed `model.json` and `classes.json`
- Confirmed the model exposes a 256 x 256 x 3 input and 128 x 128 x 31 output
- Confirmed all 5 weight shards referenced by `model.json` exist
- Served `index.html`, `model.json`, `classes.json`, and all weight shards over local HTTP

## Limitations

This is an exported inference demo, not a full research package. The repository does not currently include the training dataset, training notebook, validation metrics, or segmentation-quality benchmarks such as IoU.

## Next Improvements

- Add mean IoU, per-class IoU, and sample evaluation images
- Add a legend so each canvas color maps back to a class name
- Add before/after screenshots in this README
- Add image resizing and normalization before inference
- Add a test that runs the bundled sample image and verifies output dimensions
- Add a short model card covering dataset, preprocessing, training setup, and known failure cases
