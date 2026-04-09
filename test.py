
# Common libs
from numpy import (zeros as np_zeros, sum as np_sum, array as np_array, ceil as np_ceil,
				    floor as np_floor, max as np_max, argmax as np_argmax, int32 as np_int32,
					uint8 as np_uint8, append as np_append)
from os import makedirs
from os.path import exists, join, basename
from time import time as time_time
from argparse import ArgumentParser

# Dataset
from datasets.PLY import PLYSampler, PLYDataset, PLYCollate
from torch.utils.data import DataLoader
from torch import device as torch_device, load as torch_load, no_grad as torch_no_grad, min as torch_min
from torch.nn import Softmax as torch_nn_Softmax
from torch.cuda import is_available as torch_cuda_is_available, synchronize as torch_cuda_synchronize

from utils.config import Config
from models.architectures import Net

# PLY writer
from utils.ply import write_ply

parser = ArgumentParser()
parser.add_argument('--name', type=str, default='', help='Test name')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
parser.add_argument('--log', type=str, default='', help='Path to model.')
parser.add_argument('--model', type=str, default='', help='Chosen model.')
parser.add_argument('--infile', type=str, default='', help='PLY file to segment.')
parser.add_argument('--outdir', type=str, default='', help='Directory within which to place the output files.')
parser.add_argument('--outtype', type=str, default='colour', help='Format for output files;' \
' colour=set the colour of points to distinguish their object class;' \
' class=set a byte within the output file for each point labelling its class;' \
' class_split=split the output file into several, one for each identified object class')
parser.add_argument('--iters', type=int, default=3, help='Number of times to refine the predictions (min = 0).')

args = parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def main():

	t_tot = time_time()

	###############################
	# Choose the model to visualize
	###############################

	# args.log = join('results', args.log)
	chosen_log = args.log

	# Check if files exist
	if not exists(chosen_log):
		raise ValueError('The given log does not exists: ' + chosen_log)
	if not exists(args.infile):
		raise ValueError('The given input file does not exist: ' + args.infile)
	if not exists(args.outdir):
		raise ValueError('The given output directory does not exist: ' + args.outdir)

	############################
	# Initialize the environment
	############################

	###############
	# Previous chkp
	###############

	# Find all checkpoints in the chosen training folder
	# chkp_path = os.path.join(chosen_log, 'checkpoints')
	# chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

	# Find which snapshot to restore
	if args.model == '':
		# chosen_chkp = 'current_chkp.tar'
		chosen_chkp = 'best_chkp.tar'
	else:
		chosen_chkp = args.model
	chosen_chkp = join(chosen_log, 'checkpoints', chosen_chkp)
	print('Load pretrained: {}'.format(chosen_chkp))

	# Initialize configuration class
	config = Config()
	config.load(chosen_log)

	##################################
	# Change model parameters for test
	##################################

	# Change parameters for the test here. For example, you can stop augmenting the input data.
	#config.augment_symmetries = [False, False, False]
	#config.augment_rotation = 'none'
	#config.augment_scale_min = 0.99
	#config.augment_scale_max = 1.01
	#config.augment_noise = 0.0001
	#config.augment_color = 1.0

	#config.augment_noise = 0.0001
	#config.augment_symmetries = False
	#config.batch_num = 3
	#config.in_radius = 4
	config.validation_size = 200
	config.input_threads = 10
	

	##############
	# Prepare Data
	##############

	print()
	print('Data Preparation')
	print('****************')

	# Initiate dataset
	# if config.dataset == 'S3DIS':
	# 	test_dataset = S3DISDataset(config, args.dataset, set='validation', use_potentials=True)
	# 	test_sampler = S3DISSampler(test_dataset)
	# 	collate_fn = S3DISCollate
	if config.dataset == 'PLY':
		test_dataset = PLYDataset(config, args.infile, set='validation', use_potentials=True)
		test_sampler = PLYSampler(test_dataset)
		collate_fn = PLYCollate
	else:
		raise ValueError('Unsupported dataset : ' + config.dataset)

	# Data loader
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 sampler=test_sampler,
							 collate_fn=collate_fn,
							 num_workers=config.input_threads,
							 pin_memory=True)

	# Calibrate samplers
	test_sampler.calibration(test_loader, verbose=True)

	print('\nModel Preparation')
	print('*****************')

	# Define network model
	t1 = time_time()
	# net = Net(config, test_dataset.label_values, test_dataset.ignored_labels)
	net = Net(config, test_dataset.label_values)

	# Define a visualizer class
	tester = ModelTester(net, config, chkp_path=chosen_chkp)
	print('Done in {:.1f}s\n'.format(time_time() - t1))

	print('\nStart test')
	print('**********\n')

	# Perform the segmentation
	tester.cloud_segmentation_test(net, test_loader, config)

	print('Total Runtime: {:.1f}s\n'.format(time_time() - t_tot))



# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

	# Initialization methods
	# ------------------------------------------------------------------------------------------------------------------

	def __init__(self, net, config, chkp_path=None, on_gpu=True):

		############
		# Parameters
		############

		# Choose to train on CPU or GPU
		if on_gpu and torch_cuda_is_available():
			self.device = torch_device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))
		else:
			self.device = torch_device("cpu")
		net.to(self.device)

		# Test saving path
		self.record_file = None
		if config.saving:
			record_name = basename(chkp_path).split('.')[0]
			if args.name == '':
				self.test_path = join('test', config.saving_path.split('/')[-1], record_name)
			else:
				self.test_path = join('test', args.name, record_name)
				
			if not exists(self.test_path):
				makedirs(self.test_path)
			else:
				n = 0
				while True:
					n += 1
					new_log_dir = self.test_path + str(n)
					if not exists(new_log_dir):
						makedirs(new_log_dir)
						self.test_path = new_log_dir
						break
			print('Test path: {}'.format(self.test_path))

			if not exists(join(self.test_path, 'predictions')):
				makedirs(join(self.test_path, 'predictions'))
			
			self.record_file = open(join(self.test_path, 'TestInfo.txt'), 'w')
		else:
			self.test_path = None

		self.record(str(args))

		##########################
		# Load previous checkpoint
		##########################

		self.chkp_path = chkp_path
		checkpoint = torch_load(chkp_path, map_location='cpu')
		net.load_state_dict(checkpoint['model_state_dict'])
		self.epoch = checkpoint['epoch']
		net.eval()
		print("Model and training state restored.")

		return

	# Test main methods
	# ------------------------------------------------------------------------------------------------------------------

	def record(self, info):
		print(info)
		if self.record_file:
			self.record_file.write(info + '\n')
			self.record_file.flush()

	def cloud_segmentation_test(self, net, test_loader, config, num_votes=args.iters, debug=False):
		"""
		Test method for cloud segmentation models
		"""

		############
		# Initialize
		############

		# Choose test smoothing parameter (0 for no smoothing, 0.99 for big smoothing)
		test_smooth = 0.95
		test_radius_ratio = 0.7
		softmax = torch_nn_Softmax(1)

		# Number of classes predicted by the model
		nc_model = config.num_classes

		# Initiate global prediction over test clouds
		self.test_prob = np_zeros((test_loader.dataset.test_proj[0].shape[0], nc_model))

		#####################
		# Network predictions
		#####################

		test_epoch = 0
		last_min = -0.5

		t = [time_time()]
		last_display = time_time()
		mean_dt = np_zeros(1)

		# Start test loop
		while True:
			print('Initialize workers')
			for i, batch in enumerate(test_loader):

				# New time
				t = t[-1:]
				t += [time_time()]

				if i == 0:
					print('Done in {:.1f}s'.format(t[1] - t[0]))

				if 'cuda' in self.device.type:
					batch.to(self.device)

				# Forward pass
				with torch_no_grad():
					outputs = net(batch)

				t += [time_time()]

				# Get probs and labels
				if outputs.dim() == 1:
					outputs = outputs.unsqueeze(0)
				stacked_probs = softmax(outputs).cpu().detach().numpy()
				s_points = batch.points[0].cpu().numpy()
				lengths = batch.lengths[0].cpu().numpy()
				in_inds = batch.input_inds.cpu().numpy()
				torch_cuda_synchronize(self.device)

				# Get predictions and labels per instance
				# ***************************************

				i0 = 0
				for b_i, length in enumerate(lengths):

					# Get prediction
					points = s_points[i0:i0 + length]
					probs = stacked_probs[i0:i0 + length]
					inds = in_inds[i0:i0 + length]

					if 0 < test_radius_ratio < 1:
						mask = np_sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
						inds = inds[mask]
						probs = probs[mask]

					# Update current probs in whole cloud
					self.test_prob[inds] = test_smooth * self.test_prob[inds] + (1 - test_smooth) * probs
					i0 += length

				# Average timing
				t += [time_time()]
				if i < 2:
					mean_dt = np_array(t[1:]) - np_array(t[:-1])
				else:
					mean_dt = 0.9 * mean_dt + 0.1 * (np_array(t[1:]) - np_array(t[:-1]))

				# Display
				if (t[-1] - last_display) > 5.0:
					last_display = t[-1]
					message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
					print(message.format(test_epoch, i,
										 100 * i / config.validation_size,
										 1000 * (mean_dt[0]),
										 1000 * (mean_dt[1]),
										 1000 * (mean_dt[2])))

			# Update minimum od potentials
			new_min = torch_min(test_loader.dataset.min_potentials)
			print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
			#print([np_mean(pots) for pots in test_loader.dataset.potentials])

			# Save predicted cloud
			if last_min + 1 < new_min:

				# Update last_min
				last_min += 1

				# Save real IoU once in a while
				if int(np_ceil(new_min)) % 10 == 0 or last_min > num_votes:

					# Project predictions
					self.record('\nReproject Vote #{:d}'.format(int(np_floor(new_min))))
					t1 = time_time()

					print(0, test_loader.dataset.file, test_loader.dataset.test_proj[0].shape, self.test_prob.shape)

					print(test_loader.dataset.test_proj[0].dtype, np_max(test_loader.dataset.test_proj[0]))
					print(test_loader.dataset.test_proj[0][:5])

					# Reproject probs on the evaluations points
					proj_prob = self.test_prob[test_loader.dataset.test_proj[0], :]

					t2 = time_time()
					self.record('Done in {:.1f} s\n'.format(t2 - t1))

					# Save predictions
					print('Saving clouds')
					t1 = time_time()

					# Get the predicted labels
					preds = test_loader.dataset.label_values[np_argmax(proj_prob, axis=1)].astype(np_int32)

					# Save PLY
					if (args.outtype == 'colour'):
						# Get file
						points = test_loader.dataset.load_evaluation_points(test_loader.dataset.file)

						# Set colours
						pred_colours = np_array([test_loader.dataset.label_to_colour[i] for i in preds])

						# Print to file
						test_name = join(args.outdir, test_loader.dataset.file.split('/')[-1])
						# write_ply(test_name,
						# 		[points, pred_colours],
						# 		['x', 'y', 'z', 'red', 'green', 'blue'])
						write_ply(test_name,
								[-points[:, 0], points[:, 2], points[:, 1], pred_colours],
								['x', 'y', 'z', 'red', 'green', 'blue'])
					elif (args.outtype == 'class'):
						# Get file
						points_and_colours = test_loader.dataset.load_evaluation_points_and_colours(test_loader.dataset.file)

						# Split the points from the colours
						points = points_and_colours[:, :3]
						colours = points_and_colours[:, 3:6].astype(np_uint8)
						preds = preds.astype(np_uint8)

						# Print to file
						test_name = join(args.outdir, test_loader.dataset.file.split('/')[-1])
						# write_ply(test_name,
						# 		[points, colours, preds],
						# 		['x', 'y', 'z', 'red', 'green', 'blue', 'pred'])
						write_ply(test_name,
								[-points[:, 0], points[:, 2], points[:, 1], colours, preds],
								['x', 'y', 'z', 'red', 'green', 'blue', 'pred'])
					elif (args.outtype == 'class_split'):
						# Get file
						points_and_colours = test_loader.dataset.load_evaluation_points_and_colours(test_loader.dataset.file)

						# Sort the points and colours by class
						out_points_and_colours_list = [np_zeros((0,6)) for _ in range(13)]
						for o in range(preds.size):
							out_points_and_colours_list[preds[o]] = np_append(out_points_and_colours_list[preds[o]], [points_and_colours[o]], axis=0)

						# Write the predicted points of each class in a unique file
						for c_i in range(13):
							# Get the subset of points predicted to belong to class number c_i
							out_points = (out_points_and_colours_list[c_i])[:, :3]
							out_colours = (out_points_and_colours_list[c_i])[:, 3:6].astype(np_uint8)
		
							if (out_points.shape[0] > 0):
								# Print to file
								test_name = join(args.outdir, test_loader.dataset.label_to_names[c_i] + '_' + test_loader.dataset.file.split('/')[-1])
								# write_ply(test_name,
								# 		[out_points, out_colours],
								# 		['x', 'y', 'z', 'red', 'green', 'blue'])
								write_ply(test_name,
										[-out_points[:, 0], out_points[:, 2], out_points[:, 1], out_colours],
										['x', 'y', 'z', 'red', 'green', 'blue'])

					t2 = time_time()
					print('Done in {:.1f} s\n'.format(t2 - t1))

			test_epoch += 1

			# Break when reaching number of desired votes
			if last_min > num_votes:
				break
			else:
				print('last_min {:.1f} < num_votes {:.1f}\n'.format(last_min, num_votes))

		return


if __name__ == '__main__':
	main()
