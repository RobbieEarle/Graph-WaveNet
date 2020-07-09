from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import pandas as pd
import util

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str,
                    default='data/bach_chorales/bach_choral_set_dataset.csv', help='')
parser.add_argument('--output_dir', type=str,
                    default='garage/chorales', help='')
args = parser.parse_args()

bch_df = pd.read_csv(args.csv_path)
print("Dataset successfully read")
print("Shape:")
print(bch_df.shape)
print("Columns:")
print(bch_df.columns)
choral_ids = bch_df.choral_ID.unique()
print()

print("Generating Audio...")
for curr_id in choral_ids:
    curr_choral_df = bch_df[bch_df['choral_ID'] == curr_id]

    curr_pitches_df = curr_choral_df.iloc[:, 2:14]
    curr_pitches_df = curr_pitches_df.applymap(lambda x: 1 if x == 'YES' else 0)
    curr_pitches = curr_pitches_df.to_numpy()

    curr_velocities_df = curr_choral_df['meter'].apply(lambda x: 10 + int(x / 5 * 100))
    curr_velocities = curr_velocities_df.to_numpy()

    curr_pitches = curr_velocities.reshape((curr_velocities.shape[0], 1)) * curr_pitches
    curr_pr = curr_pitches.T
    left_padding = np.zeros((59, curr_pr.shape[1]))
    right_padding = np.zeros((57, curr_pr.shape[1]))
    padded_curr_pr = np.concatenate((left_padding, curr_pr, right_padding), axis=0)
    curr_midi = util.piano_roll_to_pretty_midi(padded_curr_pr, base_note=21)
    curr_audio = curr_midi.synthesize(fs=16000)

    print("  Saving {}: {}".format(curr_id, curr_pr.shape))
    np.save('{}/audio_{}'.format(args.output_dir, curr_id), curr_audio)

print("Done!")
