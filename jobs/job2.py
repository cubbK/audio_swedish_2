# %%
import whisper
import librosa
import uuid
from scipy.io import wavfile
from datasets import load_dataset
import torch
import os
import numpy as np
import tarfile

dataset_shard = "dataset-000000"


def transcribe_with_whisper(wav, audio_samples):
    # Resample to 16kHz which is Whisper's expected sample rate
    wav_resampled = librosa.resample(
        wav, orig_sr=audio_samples.sample_rate, target_sr=16000
    )
    # print(f"\nResampled audio:")
    # print(f"  New shape: {wav_resampled.shape}")
    # print(f"  New duration: {len(wav_resampled) / 16000:.2f} seconds")

    # Transcribe the resampled audio
    result = model.transcribe(
        wav_resampled,
        language="Swedish",
        # verbose=True,
        temperature=0,
        condition_on_previous_text=False,
        # Try without any preprocessing that might affect timing
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
    )

    segments = result["segments"]

    # Check if timestamps make sense now
    audio_duration = len(wav_resampled) / 16000
    segments_cleaned = []

    for i, segment in enumerate(segments):
        text = segment["text"].strip()  # type: ignore
        if not text:
            continue

        start_time = segment["start"]  # type: ignore
        end_time = segment["end"]  # type: ignore

        # # Debug: Print first few segments to see if timing looks reasonable
        # if i < 5:
        #     print(f"Segment {i}: {start_time:.2f}s - {end_time:.2f}s | {text[:30]}...")

        # Only keep segments within audio duration
        if start_time < audio_duration:  # type: ignore
            clipped_end = min(end_time, audio_duration)
            segments_cleaned.append(
                {
                    "start": start_time,
                    "end": clipped_end,
                    "text": text,
                }
            )

    return segments_cleaned


def write_segments_to_files(segments_cleaned, wav):
    output_dir = "audio_segments"
    os.makedirs(output_dir, exist_ok=True)

    for i, segment in enumerate(segments_cleaned):
        uid = uuid.uuid4()
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        start_sample = int(start_time * 24000)
        end_sample = int(end_time * 24000)

        segment_wav = wav[start_sample:end_sample]
        audio_segment_int16 = (segment_wav * 32767).astype(np.int16)

        os.makedirs("audio_segments", exist_ok=True)
        filename = f"segment_{uid}.wav"
        filepath = os.path.join(output_dir, filename)
        wavfile.write(filepath, 24000, audio_segment_int16)

        filename_text = f"segment_{uid}.txt"
        filepath_text = os.path.join(output_dir, filename_text)
        with open(filepath_text, "w") as f:
            f.write(text)


def create_tar_archive():
    with tarfile.open(f"{dataset_shard}.tar", "w") as tar:
        for root, dirs, files in os.walk("audio_segments"):
            for file in files:
                file_path = os.path.join(root, file)
                tar.add(file_path, arcname=file)


def upload_to_gcs():
    os.system(
        f"gsutil cp {dataset_shard}.tar gs://audio_swedish_2/jobs/{dataset_shard}.tar"
    )


# dataset_name = "cubbk/audio_swedish_2_dataset_cleaned"
# folder_name = "8sidor_audios_dataset_pairs"
# checkpoint_name = "8sidor_audios_dataset_pairs_checkpoint.txt"
# segments_until_upload = 140
# os.makedirs("archive_to_upload", exist_ok=True)

# filepath = os.path.join("audio_segments", "aaaaaaaaaaa.wav")
# audio_segment_int16 = (wav * 32767).astype(np.int16)
# wavfile.write(filepath, 24000, audio_segment_int16)

if __name__ == "__main__":
    dataset = load_dataset(
        "cubbk/audio_swedish_2_dataset_cleaned",
        split="train",
        streaming=True,
        data_dir="8sidor_audios_dataset",
        data_files={"train": f"{dataset_shard}.tar"},
    )

    model = whisper.load_model("turbo")
    for i, dataset_item in enumerate(dataset):
        print(f"Processing item {i + 1}...")
        audio_decoder = dataset_item[".wav"]  # type: ignore
        audio_samples = audio_decoder.get_all_samples()
        wav = audio_samples.data.numpy().squeeze()

        segments = transcribe_with_whisper(wav, audio_samples)
        write_segments_to_files(segments, wav)
        if i == 10:
            break

    print("Creating tar archive...")
    create_tar_archive()
    print("Uploading to GCS...")
    upload_to_gcs()
    print("Done.")

# %%
