# web-antispoof

Web-service for antispoofing detection

## Files description
* `bot.py` - runs telegram bot
* `api.py` - contains local web-server implementation for local model testing
* `inference.py` - contains feature extraction and model prediction functions + runs local test
* `example.py` - Telegram bot example

## Example usage
* To run bot:
```bash
    export https_proxy=...
    python3 bot.py
```

* To run local server:
```bash
    python3 api.py
    
    # to test the model output
     curl -i -H "Content-Type: application/json" -X POST -d '{"FILE_PATH": "/home/anton/contests/boosters/deploy/git/web-antispoof/data/test/test_25s.wav"}' 127.0.0.1:5000/predict
```

* To run model test:
```bash
    python3 inference.py
```

## Build
* Before building copy the model to `data/model`:
```bash
    cd web-antispoof/
    cp <model_path> data/model/model.h5
```

* To build Docker image:
```bash
    docker build -t web-antispoof:latest .
```

## Deploy
* To start bot-server we need to mount a folder to `/opt/audio` in container, where downloads will be stored:
```bash
    docker run -v <host_machine_path>:/opt/audio web-antispoof:latest
```

### Links
https://medium.com/@aliabdelaal/telegram-bot-tutorial-using-python-and-flask-1fc634da9522
https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions-%E2%80%93-Your-first-Bot
