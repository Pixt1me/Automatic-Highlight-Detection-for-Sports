import librosa 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import shutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Load the audio file of the sports clip.
filename = 'video.wav'  # Enter your audio file name of the match here. .wav,.mp3, etc. are supported.
vid, sample_rate = librosa.load(filename, sr=16000)

# Calculate the duration of the audio
duration_minutes = librosa.get_duration(y=vid, sr=sample_rate) / 60
print(f"Duration of audio: {duration_minutes:.2f} minutes")

# Breaking down the audio into chunks of 5 seconds to analyze energy
chunk_size = 5 
window_length = chunk_size * sample_rate

# Plotting the short time energy distribution histogram of all chunks
energy = np.array([sum(abs(vid[i:i+window_length]**2)) for i in range(0, len(vid), window_length)])
plt.hist(energy) 
plt.show()

# Setting the threshold value of commentator and audience noise above which to include portions in highlights
thresh = 300
df = pd.DataFrame(columns=['energy', 'start', 'end'])
row_index = 0
for i in range(len(energy)):
    value = energy[i]
    if value >= thresh:
        i = np.where(energy == value)[0]
        df.loc[row_index, 'energy'] = value
        df.loc[row_index, 'start'] = i[0] * chunk_size
        df.loc[row_index, 'end'] = (i[0] + 1) * chunk_size
        row_index += 1

# Merge consecutive time intervals of audio clips into one
temp = []
i, j, n = 0, 0, len(df) - 1
while i < n:
    j = i + 1
    while j <= n:
        if df['end'][i] == df['start'][j]:
            df.loc[i, 'end'] = df.loc[j, 'end']
            temp.append(j)
            j += 1
        else:
            i = j
            break  
df.drop(temp, axis=0, inplace=True)

# Extracting subclips from the video file based on the energy profile obtained from the audio file
start = np.array(df['start'])
end = np.array(df['end'])
cwd = os.getcwd()
sub_folder = os.path.join(cwd, "Subclips")
if os.path.exists(sub_folder):
    shutil.rmtree(sub_folder)
os.mkdir(sub_folder)

for i in range(len(df)):
    start_lim = max(0, start[i] - 5)  # Adjusted to ensure start_lim is not negative
    end_lim = end[i]
    filename = f"highlight{i+1}.mp4"
    ffmpeg_extract_subclip("video.mp4", start_lim, end_lim, targetname=os.path.join(sub_folder, filename))

# Concatenate the extracted highlight clips into a single video
files = [os.path.join(sub_folder, f"highlight{i+1}.mp4") for i in range(len(df))]
final_clip = concatenate_videoclips([VideoFileClip(file) for file in files])

# Write the concatenated highlight clip to a file
final_clip.write_videofile("highlights.mp4")

# Delete the temporary folder
shutil.rmtree(sub_folder)
