#!flask/bin/python
import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from threading import Lock
from typing import List, Optional, Union
from urllib.parse import parse_qs

from flask import Flask, abort, render_template, render_template_string, request, send_file

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

MAX_TEXT_LENGTH = 5000
MAX_CONTENT_LENGTH = 1 * 1024 * 1024

app = None
args = None
synthesizer = None


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("TTS_SERVER_HOST", "127.0.0.1"),
        help="address to listen on. Defaults to TTS_SERVER_HOST or 127.0.0.1; use 0.0.0.0 to expose the server.",
    )
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable verbose logging.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    return parser


def _is_within(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath([path, directory]) == directory
    except ValueError:
        return False


def style_wav_uri_to_dict(style_wav: str, allowed_dirs: Optional[List[str]] = None) -> Union[str, dict, None]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri
        allowed_dirs (List[str], optional): directories a style wav path may resolve into.

    Returns:
        Union[str, dict, None]: path to file (str) or gst style (dict)
    """
    if not style_wav:
        return None
    try:
        parsed = json.loads(style_wav)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}

    # Paths are resolved before checking so symlinks and ".." cannot escape the allowed directories.
    resolved = os.path.realpath(style_wav)
    allowed = [os.path.realpath(d) for d in (allowed_dirs or [])]
    if not any(_is_within(resolved, d) for d in allowed):
        abort(400, description="style_wav must be a GST style dict or a .wav file inside the model directory.")
    if not resolved.endswith(".wav") or not os.path.isfile(resolved):
        abort(400, description="style_wav must be a GST style dict or an existing .wav file.")
    return resolved


def _check_text_length(text: str) -> None:
    if len(text) > MAX_TEXT_LENGTH:
        abort(400, description=f"text exceeds the maximum length of {MAX_TEXT_LENGTH} characters.")


def create_app(args, synthesizer, model_path=None, config_path=None, vocoder_config_path=None) -> Flask:
    """Build the Flask app and routes around an already constructed Synthesizer."""
    use_multi_speaker = hasattr(synthesizer.tts_model, "num_speakers") and (
        synthesizer.tts_model.num_speakers > 1 or synthesizer.tts_speakers_file is not None
    )
    speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)

    use_multi_language = hasattr(synthesizer.tts_model, "num_languages") and (
        synthesizer.tts_model.num_languages > 1 or synthesizer.tts_languages_file is not None
    )
    language_manager = getattr(synthesizer.tts_model, "language_manager", None)

    # TODO: set this from SpeakerManager
    use_gst = synthesizer.tts_config.get("use_gst", False)

    allowed_style_dirs = [os.getcwd()]
    if model_path:
        allowed_style_dirs.append(os.path.dirname(os.path.abspath(model_path)))

    flask_app = Flask(__name__)
    flask_app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    if args.debug:
        flask_app.logger.setLevel(logging.DEBUG)

    lock = Lock()

    @flask_app.route("/")
    def index():
        return render_template(
            "index.html",
            show_details=args.show_details,
            use_multi_speaker=use_multi_speaker,
            use_multi_language=use_multi_language,
            speaker_ids=speaker_manager.name_to_id if speaker_manager is not None else None,
            language_ids=language_manager.name_to_id if language_manager is not None else None,
            use_gst=use_gst,
        )

    @flask_app.route("/details")
    def details():
        model_config = None
        if args.config_path is not None and os.path.isfile(args.config_path):
            model_config = load_config(args.config_path)
        elif args.model_name is not None and config_path is not None:
            model_config = load_config(config_path)

        if args.vocoder_config_path is not None and os.path.isfile(args.vocoder_config_path):
            vocoder_config = load_config(args.vocoder_config_path)
        elif args.vocoder_name is not None and vocoder_config_path is not None:
            vocoder_config = load_config(vocoder_config_path)
        else:
            vocoder_config = None

        return render_template(
            "details.html",
            show_details=args.show_details,
            model_config=model_config,
            vocoder_config=vocoder_config,
            args=args.__dict__,
        )

    @flask_app.route("/api/tts", methods=["GET", "POST"])
    def tts():
        with lock:
            text = request.headers.get("text") or request.values.get("text", "")
            speaker_idx = request.headers.get("speaker-id") or request.values.get("speaker_id", "")
            language_idx = request.headers.get("language-id") or request.values.get("language_id", "")
            style_wav = request.headers.get("style-wav") or request.values.get("style_wav", "")
            _check_text_length(text)
            style_wav = style_wav_uri_to_dict(style_wav, allowed_style_dirs)

            print(f" > Model input length: {len(text)}")
            print(f" > Speaker Idx: {speaker_idx}")
            print(f" > Language Idx: {language_idx}")
            wavs = synthesizer.tts(text, speaker_name=speaker_idx, language_name=language_idx, style_wav=style_wav)
            out = io.BytesIO()
            synthesizer.save_wav(wavs, out)
        return send_file(out, mimetype="audio/wav")

    # Basic MaryTTS compatibility layer

    @flask_app.route("/locales", methods=["GET"])
    def mary_tts_api_locales():
        """MaryTTS-compatible /locales endpoint"""
        # NOTE: We currently assume there is only one model active at the same time
        if args.model_name is not None:
            model_details = args.model_name.split("/")
        else:
            model_details = ["", "en", "", "default"]
        return render_template_string("{{ locale }}\n", locale=model_details[1])

    @flask_app.route("/voices", methods=["GET"])
    def mary_tts_api_voices():
        """MaryTTS-compatible /voices endpoint"""
        # NOTE: We currently assume there is only one model active at the same time
        if args.model_name is not None:
            model_details = args.model_name.split("/")
        else:
            model_details = ["", "en", "", "default"]
        return render_template_string(
            "{{ name }} {{ locale }} {{ gender }}\n", name=model_details[3], locale=model_details[1], gender="u"
        )

    @flask_app.route("/process", methods=["GET", "POST"])
    def mary_tts_api_process():
        """MaryTTS-compatible /process endpoint"""
        with lock:
            if request.method == "POST":
                data = parse_qs(request.get_data(as_text=True))
                # NOTE: we ignore param. LOCALE and VOICE for now since we have only one active model
                text = data.get("INPUT_TEXT", [""])[0]
            else:
                text = request.args.get("INPUT_TEXT", "")
            _check_text_length(text)
            print(f" > Model input length: {len(text)}")
            wavs = synthesizer.tts(text)
            out = io.BytesIO()
            synthesizer.save_wav(wavs, out)
        return send_file(out, mimetype="audio/wav")

    return flask_app


def build_app(parsed_args) -> Flask:
    """Resolve model paths, construct the Synthesizer and the Flask app for the parsed CLI arguments."""
    global app, args, synthesizer  # pylint: disable=global-statement
    args = parsed_args

    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path)

    # CASE1: list pre-trained TTS models
    if args.list_models:
        manager.list_models()
        sys.exit()

    # update in-use models to the specified released models.
    model_path = None
    config_path = None
    speakers_file_path = None
    vocoder_path = None
    vocoder_config_path = None

    # CASE2: load pre-trained model paths
    if args.model_name is not None and not args.model_path:
        model_path, config_path, model_item = manager.download_model(args.model_name)
        args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

    if args.vocoder_name is not None and not args.vocoder_path:
        vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

    # CASE3: set custom model paths
    if args.model_path is not None:
        model_path = args.model_path
        config_path = args.config_path
        speakers_file_path = args.speakers_file_path

    if args.vocoder_path is not None:
        vocoder_path = args.vocoder_path
        vocoder_config_path = args.vocoder_config_path

    # load models
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        tts_speakers_file=speakers_file_path,
        tts_languages_file=None,
        vocoder_checkpoint=vocoder_path,
        vocoder_config=vocoder_config_path,
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=args.use_cuda,
    )

    app = create_app(
        args,
        synthesizer,
        model_path=model_path,
        config_path=config_path,
        vocoder_config_path=vocoder_config_path,
    )
    return app


def main():
    parsed_args = create_argparser().parse_args()
    flask_app = build_app(parsed_args)
    # The Werkzeug debugger and reloader stay off so --debug never exposes an interactive console.
    flask_app.run(host=parsed_args.host, port=parsed_args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
