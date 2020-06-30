# Serving TensorFlow with Actix-Web

This repository gives an example of training a machine learning model using TensorFlow2.0 Keras in python, exporting that model, then serving inference over a RESTful API using Actix-Web in rust.


The motivation behind this example was to try out the [TensorFlow rust bindings](https://github.com/tensorflow/rust) in a simple, practical use case.


For more information on the tools used, check out the following repositories:
* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [TensorFlow Rust](https://github.com/tensorflow/rust)
* [Actix Web](https://github.com/actix/actix-web)


## Overview
The repository has 3 sections:
* `./training` – Contains the script which trains a neural network to recognize digits.
* `./server` – Contains the RESTful API webserver rust code.
* `./client` – Contains a sample script that demonstrates making a request to the server.

The training script will output a saved neural network model in TensorFlow's protobuf format. The server then loads this into memory on startup. The server accepts a JSON payload at the `/mnist` endpoint with a single key "image" that is a base64 encoded image (PNG or JPG). This image is decoded, rescaled to the correct input dimensions, converted to grayscale, normalized (matching the training data normalization), and finally submitted to the model for inference. Predictions are returned with a "label" integer value and a "confidence" float between 0 and 1.

## Setup
This example assumes you have [rust installed](https://www.rust-lang.org/tools/install) and python 3.6+ setup. To install the needed python dependencies:
```
pip install -r requirements.txt
```

## Training
The model used is a simple convolutional neural network trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The data is automatically downloaded using the `tensorflow-datasets` library. To train the model:
```
python training/train.py
```
This will output a saved model to `./saved_model`. A pre-trained model is included in this repository. The model isn't too large and can be trained without any GPU.

## Serving
The server code is a rust crate located in `./server`. In order to run, the server requires the saved model directory location specified with the `--model-dir` command line argument. you can try running the server with:
```
cd server
cargo run -- --model-dir ../saved_model
```

## Serving in Docker
For actual deployments, you probably would want to build a release in a container to serve the API. To build the docker image:
```
docker build -t actix-tf .
```
Then to run the image locally for testing:
```
docker run --rm -it -p 8080:8080 actix-tf
```

## Client Testing
With the server running locally, you can test inference using `./client/client.py`. Included is a PNG file with a handwritten "3" that is base64-encoded and submitted to the server.

To test:
```
python client/client.py
```

Input:

<img src="client/test_image_3.png" width="80">

Expected output:
```
POST to http://127.0.0.1:8080/mnist
Response (200)
Content: {
    "label": 3,
    "confidence": 0.9999999
}
```
