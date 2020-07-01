from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import time
import pretty_midi
import pandas as pd
import util


def generate_graph_seq2seq_io_data(
        midi_data, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = midi_data.shape
    midi_data = np.expand_dims(midi_data, axis=-1)
    midi_data = midi_data.astype(int)
    # x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    x = np.zeros((max_t - min_t, len(x_offsets), num_nodes, 1), dtype='uint8')
    y = np.zeros((max_t - min_t, len(y_offsets), num_nodes, 1), dtype='uint8')
    # print(x.shape)
    # print(min_t, max_t)
    # print("asdfs"+234)
    for t in range(min_t, max_t):  # t is the index of the last observation.

        x[t - min_t] = midi_data[t + x_offsets, ...]
        y[t - min_t] = midi_data[t + y_offsets, ...]

        # x.append(midi_data[t + x_offsets, ...])
        # y.append(midi_data[t + y_offsets, ...])
        # if t == 2000:
        #     print()
        #     print(x_offsets)
        #     print(y_offsets)
        #     print(t)
        #     print(t + x_offsets)
        #     print(t + y_offsets)
        #     print()
        #     print(midi_data.shape)
        #     print(midi_data[t + x_offsets, ...].shape)
        #     print(midi_data[t + y_offsets, ...].shape)
    # x = np.stack(x, axis=0)
    # y = np.stack(y, axis=0)
    # print(x.shape)
    # print(x[2500, 0, :, 0])
    # time.sleep(1)
    # print("sfda"+234)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    midi_data = np.transpose(pretty_midi.PrettyMIDI(args.training_song_filename).get_piano_roll(fs=args.fs))

    print(midi_data.shape)
    print(midi_data[1000])
    print()

    # f = open('data/bach_chorales/chorales_csv/chorales.lisp')
    # file = f.read()
    # splitfile = file.split('\n')
    # print(len(splitfile))
    # print(splitfile[0])

    bch_df = pd.read_csv('data/bach_chorales/chorales_csv/bach_choral_set_dataset.csv')
    print(bch_df.shape)
    print(bch_df.columns)
    choral_ids = bch_df.choral_ID.unique()
    for curr_id in choral_ids:
        curr_choral_df = bch_df[bch_df['choral_ID'] == curr_id]

        curr_pitches_df = curr_choral_df.iloc[:, 2:14]
        curr_pitches_df = curr_pitches_df.applymap(lambda x: 1 if x == 'YES' else 0)
        # print(curr_pitches_df.shape)
        curr_pitches = curr_pitches_df.to_numpy()
        # print(curr_pitches.shape)

        curr_velocities_df = curr_choral_df['meter'].apply(lambda x: 10 + int(x / 5 * 100))
        curr_velocities = curr_velocities_df.to_numpy()
        # print(curr_velocities.shape)
        # print(curr_velocities)
        # print(curr_pitches.shape)
        # print(curr_pitches[:10, :])

        curr_pitches = curr_velocities.reshape((curr_velocities.shape[0], 1)) * curr_pitches
        curr_pr = curr_pitches.T
        left_padding = np.zeros((59, curr_pr.shape[1]))
        right_padding = np.zeros((37, curr_pr.shape[1]))
        curr_pr = np.concatenate((left_padding, curr_pr, right_padding), axis=0)
        curr_midi = util.piano_roll_to_pretty_midi(curr_pr, 1)
        curr_audio = curr_midi.synthesize(fs=16000)
        print("Saving " + str(curr_id))
        np.save('garage/testing_chorales/audio_{}'.format(curr_id), curr_audio)

        # print(curr_pitches.shape)
        # print(curr_pitches[:10, :])
        #
        # time.sleep(1)
        # print("asdf" + 1234)

    # print(midi_data.shape)
    time.sleep(1)
    print("asdf"+1234)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        midi_data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/song_data", help="Output directory.")
    parser.add_argument("--training_song_filename", type=str, default="data/selected_piano/beethoven_tempest.midi", help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=500, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=500, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)
    parser.add_argument("--fs", type=int, default=20, help="Samples our song every 1/fs of a second",)

    args = parser.parse_args()
    # if os.path.exists(args.output_dir):
    #     reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
    #     if reply[0] != 'y': exit
    # else:
    #     os.makedirs(args.output_dir)
    generate_train_val_test(args)
