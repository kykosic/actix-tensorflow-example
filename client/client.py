#!/usr/bin/env python
"""
    Example python client for testing the prediction server
"""
import base64
import json
import os

import requests


if __name__ == '__main__':
    url = "http://127.0.0.1:8080/mnist"
    this_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(this_dir, 'test_image_3.png')

    print(f"Reading image from {image_path}")
    with open(image_path, 'rb') as f:
        image = f.read()

    data = {
        'image': base64.b64encode(image).decode()
    }
    print(f"POST to {url}")
    res = requests.post(url,
                        json=data,
                        headers={'content-type': 'application/json'})
    print(f"Response ({res.status_code})")
    if res.status_code == 200:
        print(f"Content: {json.dumps(res.json(), indent=4)}")
    else:
        print(f"Body: {res.text}")
