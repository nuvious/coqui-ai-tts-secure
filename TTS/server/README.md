# :frog: TTS demo server
Before you use the server, make sure you [install](https://github.com/coqui-ai/TTS/tree/dev#install-tts)) :frog: TTS properly. Then, you can follow the steps below.

**Note:** If you install :frog:TTS using ```pip```, you can also use the ```tts-server``` end point on the terminal.

**Note:** The server listens on 127.0.0.1 by default; pass `--host 0.0.0.0` (or set `TTS_SERVER_HOST`) to expose it to other machines. The Docker images already set `TTS_SERVER_HOST=0.0.0.0`.

Examples runs:

List officially released models.
```python TTS/server/server.py  --list_models ```

Run the server with the official models.
```python TTS/server/server.py  --model_name tts_models/en/ljspeech/tacotron2-DCA --vocoder_name vocoder_models/en/ljspeech/multiband-melgan```

Run the server with the official models on a GPU.
```CUDA_VISIBLE_DEVICES="0" python TTS/server/server.py  --model_name tts_models/en/ljspeech/tacotron2-DCA --vocoder_name vocoder_models/en/ljspeech/multiband-melgan --use_cuda True```

Run the server with a custom models.
```python TTS/server/server.py  --tts_checkpoint /path/to/tts/model.pth --tts_config /path/to/tts/config.json --vocoder_checkpoint /path/to/vocoder/model.pth --vocoder_config /path/to/vocoder/config.json```
