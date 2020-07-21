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
        pr_data, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
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

    num_samples, num_nodes = pr_data.shape
    midi_data = np.expand_dims(pr_data, axis=-1)
    midi_data = midi_data.astype(int)
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    x = np.zeros((max_t - min_t, len(x_offsets), num_nodes, 1), dtype='uint8')
    y = np.zeros((max_t - min_t, len(y_offsets), num_nodes, 1), dtype='uint8')
    for t in range(min_t, max_t):  # t is the index of the last observation.

        x[t - min_t] = midi_data[t + x_offsets, ...]
        y[t - min_t] = midi_data[t + y_offsets, ...]

    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y

    if args.dataset == 'maestro':
        pr_data = np.transpose(pretty_midi.PrettyMIDI(args.raw_data_path).get_piano_roll(fs=args.fs))

    elif args.dataset == 'bch':

        bch_df = pd.read_csv(args.raw_data_path)
        pitches_df = bch_df.iloc[:, 2:14]
        pitches_df = pitches_df.applymap(lambda x: 1 if x == 'YES' else 0)
        pr_data = pitches_df.to_numpy()
        if args.velocities:
            velocities_df = bch_df['meter'].apply(lambda x: 10 + int(x / 5 * 100))
            velocities = velocities_df.to_numpy()
            velocities = np.expand_dims(velocities, axis=1)
            pr_data = velocities * pr_data

    print(pr_data.shape)
    print(pr_data)

    if args.num_train_samples > 0:
        pr_data = pr_data[:args.num_train_samples, :]

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        pr_data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )

    print(x.shape)
    print('x: {}'.format(np.transpose(x, (0, 2, 1, 3))[:5, :, :, 0]))
    print('y: {}'.format(np.transpose(y, (0, 2, 1, 3))[:5, :, :, 0]))
    print("324"+1234)

    if args.num_train_samples > 0:
        x = np.tile(x, (args.num_sample_duplicates, 1, 1, 1))
        y = np.tile(y, (args.num_sample_duplicates, 1, 1, 1))

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    num_samples = x.shape[0]
    if args.only_train:
        num_test = 1
        num_val = 1
    else:
        num_test = round(num_samples * 0.2)
        num_val = round(num_samples * 0.1)
    num_train = num_samples - num_test - num_val
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    datasets = ["train", "val", "test"]

    # print(x_train.shape)
    # print(x[0, :, :, 0])

    print("x: {}".format(x_train[:10, 0, :, 0]))
    print("y: {}".format(y_train[:10, 0, :, 0]))

    for cat in datasets:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # print(_x[2347, :, :, 0])
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
    parser.add_argument("--output_dir", type=str, default="data/bch_data", help="Output directory.")
    parser.add_argument("--raw_data_path", type=str, default="data/bach_chorales/bach_choral_set_dataset.csv", help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)
    parser.add_argument("--only_train", action='store_true', )
    parser.add_argument("--fs", type=int, default=1, help="Samples our song every 1/fs of a second",)
    parser.add_argument("--dataset", type=str, default="bch", help="Which dataset to use. Supports bch or maestro")
    parser.add_argument("--velocities", action='store_true')
    parser.add_argument("--num_train_samples", type=int, default=0, help="How many training samples to use", )
    parser.add_argument("--num_sample_duplicates", type=int, default=100, help="How many training samples to use", )

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
