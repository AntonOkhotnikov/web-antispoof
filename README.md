# web-antispoof

Web-service for antispoofing detection

## Files description
* `bot.py` - runs telegram bot
* `api.py` - contains local web-server implementation for local model testing
* `inference.py` - contains feature extraction and model prediction functions + runs local test
* `example.py` - Telegram bot example

## Example usage
* Configure two variables in `bot.py` first:
```bash
    # Telegram bot token
    TOKEN = '<YOUR_TELEGRAM_BOT_TOKEN_GOES_HERE>'

    # proxy server
    REQUEST_KWARGS = {
        'proxy_url': 'http://USERNAME:PASSWORD@YOUR_IP_ADDRESS:PORT',
}
```

* To run bot:
```bash
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

* To run remote API:
```bash
    # on a remote machine
    python3 api_remote.py

    # to make a request from the same machine
    python3 test_request.py

    # to make a remote request change the "url" variable in test_request.py
```

## Build
* To build Docker image:
```bash
    docker build -t web-antispoof:latest .
```

## Deploy
* To start bot-server we need to mount a folder to `/opt/audio` in container, where downloads will be stored:
```bash
    docker run --rm -v <host_machine_path>:/opt/audio web-antispoof:latest
```

* To run API:
```bash
    # select another port if it is not 5000
    docker run --rm -p 5000:5000 web-antispoof:latest python3 api_remote.py
```

### Links
* https://medium.com/@aliabdelaal/telegram-bot-tutorial-using-python-and-flask-1fc634da9522
* https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions-%E2%80%93-Your-first-Bot
