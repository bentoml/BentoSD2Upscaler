This document demonstrates how to build an image generation application with upscaling ability using BentoML, powered by [diffusers](https://github.com/huggingface/diffusers).

## **Prerequisites**

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/latest/get-started/quickstart.html) first.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See Installation for details.
- If you want to test this service locally, we highly recommend using a Nvidia GPU with more than 16G VRAM.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

2024-01-19T06:16:28+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:SD2Service" listening on http://localhost:3000 (Press CTRL+C to quit)
Loading pipeline components...: 100%
```

The server is now active at [http://0.0.0.0:3000](http://0.0.0.0:3000/). You can interact with it using Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
  'http://localhost:3000/txt2img' \
  -H 'accept: image/*' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "photo of a majestic sunrise in the mountains, best quality, 4k",
  "negative_prompt": "low quality, bad quality, sketches",
  "height": 320,
  "width": 320,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "upscale": true
}'
```

## Deploy the application to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. A configuration YAML file (`bentofile.yaml`) is used to define the build options for your application. It is used for packaging your application into a Bento. See [Bento build options](https://docs.bentoml.com/en/latest/concepts/bento.html#bento-build-options) to learn more.

Make sure you have logged in to BentoCloud, then run the following command in your project directory to deploy the application to BentoCloud. Under the hood, this commands automatically builds a Bento, push the Bento to BentoCloud, and deploy it on BentoCloud.

```bash
bentoml deploy .
```

**Note**: Alternatively, you can manually build the Bento, containerize the Bento as a Docker image, and deploy it in any Docker-compatible environment. See [Docker deployment](https://docs.bentoml.org/en/latest/concepts/deploy.html#docker) for details.

Once the application is up and running on BentoCloud, you can access it via the exposed URL.
