from flask import Flask
from transformers import pipeline
import torch

app = Flask(__name__)

@app.route('/')
def hello_world():
   return "Hello, World!"

@app.route('/audio')
def audio_summary():
   device = "cuda:0" if torch.cuda.is_available() else "cpu"
   transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium", chunk_length_s=1, device=device)
   transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")
   audioURL = "https://kesav-dev-dev-ed.file.force.com/sfc/dist/version/download/?oid=00D2w000008WU8p&ids=0682w00000VnlqQ&d=%2Fa%2F2w000000JRd5%2F_uvHKhdl4zvajmOG7wZi.aNSZp5l2rkeSvt2PcV_Qfk&asPdf=false"
   res = transcribe(audioURL)["text"]
   print('Transcription: ', res)
   return res

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000)