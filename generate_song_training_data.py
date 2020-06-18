from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import time
import pretty_midi


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
    # print(midi_data.shape)
    # time.sleep(2)
    # print("asdf" + 234)
    #
    # # num_samples, num_nodes = df.shape
    # # data = np.expand_dims(df.values, axis=-1)
    # # print(data.shape)
    # # feature_list = [data]
    # # if add_time_in_day:
    # #     time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    # #     time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    # #     feature_list.append(time_in_day)
    # # if add_day_in_week:
    # #     dow = df.index.dayofweek
    # #     dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
    # #     feature_list.append(dow_tiled)
    #
    # data = np.concatenate(feature_list, axis=-1)
    # print(type(data))
    # time.sleep(2)
    # print("asdf"+234)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(midi_data[t + x_offsets, ...])
        y.append(midi_data[t + y_offsets, ...])
        # if t == 11:
        #     print()
        #     print(x_offsets)
        #     print(y_offsets)
        #     print(t)
        #     print(t + x_offsets)
        #     print(t + y_offsets)
        #     print()
        #     print(data.shape)
        #     print(data[t + x_offsets, ...].shape)
        #     print(data[t + y_offsets, ...].shape)
        #     print(data[:12, 0, :] - data[t + x_offsets, 0, :])
        #     print()
        #     print(data[12:24, 0, :] - data[t + y_offsets, 0, :])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    # print(x.shape)
    # print(y.shape)
    # time.sleep(2)
    # print("asdf" + 234)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    # df = pd.read_hdf(args.traffic_df_filename)
    midi_data = np.transpose(pretty_midi.PrettyMIDI(args.training_song_filename).get_piano_roll())
    # print(df)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # print(x_offsets)
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # print(y_offsets)
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
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
