mkdir -p 8sidor_audios

# Convert to 24000 Hz mono for your model
convert_file() {
    input="$1"
    output="8sidor_audios/$(basename "${input%.mp3}.wav")"
    # -ar 24000: sample rate, -ac 1: mono (1 channel)
    ffmpeg -i "$input" -ar 24000 -ac 1 "$output" -loglevel quiet
}

export -f convert_file

# Use all CPU cores with progress bar
ls 8_sidor_audios_full/*.mp3 | parallel --progress convert_file