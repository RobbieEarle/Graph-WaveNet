import util
import argparse
from model import *
import numpy as np
import pretty_midi
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='transition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', type=str, help='')
parser.add_argument("--training_song_filename", type=str, default="data/selected_piano/beethoven_tempest.midi",
                    help="Raw traffic readings.")
parser.add_argument("--fs", type=int, default=20, help="Samples our song every 1/fs of a second")
parser.add_argument("--sample_time", type=int, default=0, help="Sets when we start our sample")
parser.add_argument('--dataset', type=str, default='bch', help='Which dataset was used for training')
parser.add_argument("--raw_data_path", type=str, default="data/bach_chorales/bach_choral_set_dataset.csv",
                    help="Raw traffic readings.")
parser.add_argument("--choral_ID", type=str, default="000408b_",
                    help="Which choral to use for testing")

args = parser.parse_args()


def pad_choral(choral):
    left_padding = np.zeros((59, choral.shape[1]))
    right_padding = np.zeros((57, choral.shape[1]))
    return np.concatenate((left_padding, choral, right_padding), axis=0)


def main():
    if args.dataset == 'maestro':
        frequencies = np.array(
            [8.176, 8.662, 9.177, 9.723, 10.301, 10.913, 11.562, 12.250, 12.978, 13.750, 14.568, 15.434, 16.352, 17.324,
             18.354, 19.445, 20.601, 21.826, 23.124, 24.499, 25.956, 27.500, 29.135, 30.867, 32.703, 34.648, 36.708,
             38.890, 41.203, 43.653, 46.249, 48.999, 51.913, 55.000, 58.270, 61.735, 65.406, 69.295, 73.416, 77.781,
             82.406, 87.307, 92.499, 97.998, 103.82, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81,
             174.61, 184.99, 195.99, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
             369.99, 391.99, 415.31, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99,
             783.99, 830.61, 880.00, 932.32, 987.77, 1046.5, 1108.7, 1174.7, 1244.5, 1318.5, 1396.9, 1480.0, 1568.0,
             1661.2, 1760.0, 1864.7, 1975.5, 2093.0, 2217.5, 2349.3, 2489.0, 2637.0, 2793.8, 2960.0, 3136.0, 3322.4,
             3520.0, 3729.3, 3951.1, 4186.0, 4434.9, 4698.6, 4978.0, 5274.0, 5587.7, 5919.9, 6271.9, 6644.9, 7040.0,
             7458.6, 7902.1, 8372.0, 8869.8, 9397.3, 9956.1, 10548.1, 11175.3, 11839.8, 12543.9])
        piano_adj = np.zeros((128, 128))
        for row in range(128):
            piano_adj[row] = frequencies - frequencies[row]

    else:
        positions = np.arange(12)
        # frequencies = np.array(
        #     [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 391.99, 415.31, 440.00, 466.16, 493.88])
        piano_adj = np.zeros((12, 12))
        for row in range(12):
            piano_adj[row] = np.abs(positions - positions[row])

        print("B_ch adj mat, \n"+str(piano_adj))

    device = torch.device(args.device)
    adj_mx = util.load_piano_adj(piano_adj, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    print("Generating model... ")
    model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj,
                  aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid,
                  dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16)
    print("  Done")
    model.to(device)
    print("Loading state " + str(args.checkpoint) + "...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(device)))
    print("  Done")
    model.eval()

    print("Loading sample song " + str(args.training_song_filename) + "...")
    if args.dataset == 'maestro':
        midi_data = pretty_midi.PrettyMIDI(args.training_song_filename)
        pr_data = midi_data.get_piano_roll(fs=20)
    elif args.dataset == 'bch':
        bch_df = pd.read_csv(args.raw_data_path)
        bch_df = bch_df[bch_df['choral_ID'] == args.choral_ID]
        pitches_df = bch_df.iloc[:, 2:14]
        pitches_df = pitches_df.applymap(lambda x: 1 if x == 'YES' else 0)
        pitches = pitches_df.to_numpy()
        velocities_df = bch_df['meter'].apply(lambda x: 10 + int(x / 5 * 100))
        velocities = velocities_df.to_numpy()
        velocities = np.expand_dims(velocities, axis=1)
        pr_data = np.transpose(velocities * pitches)
        # print(type(pr_data))
        # print(pr_data.shape)
    elif args.dataset == 'single':
        print("Testing single pressing (no lifting)")
        pr_data = np.zeros((12, 101))

        pr_data[3, :] = 1
    elif args.dataset == 'single_p':
        print("Testing single pressing (lifting)")
        pr_data = np.zeros((12, 101))

        for i in range(101):
            pr_data[3, i] = 70

    elif args.dataset == 'press_lift':
        print("Testing music: press 2 sec, lift 2 sec, iterate")
        pr_data = np.zeros((12, 101))
        for i in range(101):
            if i %4 < 2:
                pr_data[3, i] = 1
            else:
                pr_data[4, i] = 1

    elif args.dataset == 'dual_press':
        print("Testing: Dual pressing")
        pr_data = np.zeros((12, 101))
        pr_data[3, :] = 1
        pr_data[8, :] = 1

    elif args.dataset == 'bounce':
        print("Testing: Bouncing")
        pr_data = np.zeros((12, 101))
        for i in range(101):
            m = i % 4
            if m == 0:
                pr_data[3, i] = 1
            elif m == 1 or m == 3:
                pr_data[4, i] = 1
            elif m == 2:
                pr_data[5, i] = 1

    elif args.dataset == 'dual_bounce_diff_period':
        print("Testing: Dual Bouncing with different periods")
        pr_data = np.zeros((12, 101))
        for i in range(101):
            m = i % 7
            n = i % 11
            if m == 0:
                pr_data[0,i] = 1
            elif m == 1 or m == 6:
                pr_data[1,i] = 1
            elif m == 2 or m == 5:
                pr_data[2,i] = 1
            elif m == 3 or m == 4:
                pr_data[3,i] = 1

            if n == 0:
                pr_data[4,i] = 1
            elif n == 1 or n == 10:
                pr_data[5,i] = 1
            elif n == 2 or n == 9:
                pr_data[6,i] = 1
            elif n == 3 or n == 8:
                pr_data[7,i] = 1
            elif n == 4 or n == 7:
                pr_data[8,i] = 1
            elif n == 5 or n == 6:
                pr_data[9,i] = 1

    pr_sample = pr_data[:, args.sample_time:args.sample_time + args.seq_length]
    pr_sample_label = pr_data[:, args.sample_time + args.seq_length:args.sample_time + (2 * args.seq_length)]

    print("  Done")

    print("Generating prediction...")
    model_in = pr_sample.T
    model_in = np.reshape(model_in, (1, model_in.shape[0], model_in.shape[1], 1))
    model_in = torch.Tensor(model_in).to(device)
    model_in = model_in.transpose(1, 3)
    with torch.no_grad():
        preds = model(model_in).transpose(1, 3)
    print("  Done")

    print("Synthesizing audio...")
    prediction = preds.squeeze().cpu().numpy()
    padded_prediction = pad_choral(prediction)
    pred_midi_sample = util.piano_roll_to_pretty_midi(padded_prediction, 1)
    generated_audio = pred_midi_sample.synthesize(fs=16000)

    padded_sample = pad_choral(pr_sample)
    midi_sample = util.piano_roll_to_pretty_midi(padded_sample, 1)
    sample_audio = midi_sample.synthesize(fs=16000)

    padded_sample_label = pad_choral(pr_sample_label)
    midi_sample_label = util.piano_roll_to_pretty_midi(padded_sample_label, 1)
    sample_label_audio = midi_sample_label.synthesize(fs=16000)
    print("  Done")
    garage_name = args.checkpoint.split('/')[-1]
    print("Saving data...")

    np.save('MODEL_audio_sample', sample_audio)
    np.save('MODEL_audio_sample_label', sample_label_audio)
    np.save('MODEL_audio_generated', generated_audio)
    np.save('MODEL_pr_sample', pr_sample)
    np.save('MODEL_pr_sample_label', pr_sample_label)
    np.save('MODEL_pr_generated', prediction)
    print("  Done")
    # +str(garage_name)



if __name__ == "__main__":
    main()
