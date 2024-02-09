# gradio-ASL-fingerspelling-recognition

This code is for a kaggle compeition for American Sign Language fingerspelling recognition. Refer to the [competition page](https://www.kaggle.com/competitions/asl-fingerspelling?rvi=1) for more information about the competition.

This repository contains the code to perform american sign language fingerspelling recognition on input from your Webcam or by uploading a video(with a Gradio interface). 


## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Usage](#usage)

## Prerequisites

- Docker (recommended)
- Python version 3.8.17

## Setup

1. Clone the repository

    ```
    git clone https://github.com/SamratThapa120/gradio-ASL-fingerspelling-recognition.git
    ```

2. Enter the directory

    ```
    cd gradio-ASL-fingerspelling-recognition
    ```
   
3. If you're using Docker, use the following commands:

    ```
    docker pull python:3.8.17
    docker run -it --rm -v "$(pwd)":/app -w /app python:3.8.17 bash
    ```
    This will setup a Docker container with Python 3.8.17 and map your current directory into the Docker container.

4. Install the requirements

    ```
    pip install -r requirements.txt
    ```

5. Place your `model.tflite` and `inference_args.json` under the `weights` directory

## Running the App

To start the Gradio application, use the following command:

```bash
python3 app.py
```

This command will start a local server that hosts the application. You can then access the app by visiting http://localhost:7860/ in your web browser.

## UI Preview

![github](https://github.com/SamratThapa120/gradio-ASL-fingerspelling-recognition/assets/38401989/bfbed9fb-0084-480f-af6a-76505889af55)



