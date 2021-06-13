from flask import Flask, app, request, send_from_directory
from scipy.io.wavfile import write
from scipy.signal import butter, sosfilt
import noisereduce as nr
import librosa

app = Flask(__name__)


def butter_bandpass_filter(datas, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfilt(sos, datas)
    return y


def process(files):
    data, sr = librosa.load(files[0])
    noisypart, sr1 = librosa.load(files[1])
    datax = nr.reduce_noise(audio_clip=data, noise_clip=noisypart, verbose=False)
    lung = butter_bandpass_filter(datax, 100, 2500, sr)
    path = r"D:\lungSound.wav"
    write(path, sr, lung)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        files = request.files.getlist('file')
        process(files)
    if request.method == 'GET':
        return send_from_directory("D:", "lungSound.wav", as_attachment=True)
    return 'BP Filter API'


if __name__ == '__main__':
    app.run()
