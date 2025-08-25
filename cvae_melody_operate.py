import tensorflow as tf
from cvae_melody_utils import make_condition_vector, generate_waveform
from cvae_melody import CVAE, Sampling
import mido

# 学習済みモデル読み込み
cvae = tf.keras.models.load_model(
    "cvae_model.keras",
    compile=False,
    custom_objects={"CVAE": CVAE, "Sampling": Sampling},
)

# 条件ベクトルを作成
cond_vec = make_condition_vector(
    pitch=2,
    waveform_type=0,
    layer_waveform=0,
    thickness=0.1,
    brightness=1.0,
    distortion=0.0,
    delay_reverb=0.3,
    texture=0.0,
    lowpass=0.0,
    highpass=1.0,
    sweep_close=0.0,
    sweep_open=0.0,
    attack=1.0,
    release=0.0,
    side=0.0,
)

waveform = generate_waveform(cvae, cond_vec)


# 利用可能なMIDI入力ポートを表示
print("Available MIDI input ports:")
for i, port in enumerate(mido.get_input_names()):
    print(f"{i}: {port}")

# 使いたいMIDI入力ポートを選択（例: 0番目）
port_index = 0
port_name = mido.get_input_names()[port_index]

print(f"\nUsing input port: {port_name}")


# MIDIノート番号を音名に変換する関数
def note_number_to_name(note: int) -> str:
    note_names = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]
    octave = (note // 12) - 1  # MIDI標準では60がC4
    name = note_names[note % 12]
    return f"{name}{octave}"


# 入力を監視してノートを表示
with mido.open_input(port_name) as inport:
    print("Listening for MIDI input... (Ctrl+C to stop)")
    for msg in inport:
        if msg.type == "note_on" and msg.velocity > 0:
            note_number = msg.note
            note_name = note_number_to_name(note_number)
            print(f"Note ON : {note_number} ({note_name})")
        elif msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        ):
            note_number = msg.note
            note_name = note_number_to_name(note_number)
            print(f"Note OFF: {note_number} ({note_name})")
