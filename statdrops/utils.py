# Copyright 2020 Xiaochen Zheng @ETHZ and Jörg Rieckermann @EAWAG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file includes necessary operations for accessing to the SQL database by Python as well as
# the format converting between .db and .csv. Most of these functions are originally developed by
# the authors otherwise the sources are mentioned.
# ==============================================================================

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from tqdm import tqdm
import time
import math

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches


class EventSplitter(object):  # TODO save different events to npy file
	"""docstring for ClassName"""
	def __init__(self, path, file_name,):
		super(EventSplitter, self).__init__()
		self.path = path
		self.file_name = file_name

	def raintime_extractor(self, intensity_pos, num_col, eps=1.2):
		"""Extract the time points of raining based on the rain intensity.

		Args:
			intensity_pos: the column position of rain intensity in a matrix, from zero to (num_col-1)
			num_col: the number of column
			eps: the threshold value to identify whether it's raining or not [mm/h]

		Returns:
			np.ndarray containing all time points of raining whose shape is the same with data_array

		"""
		data_array = np.load(self.path+self.file_name, allow_pickle=True)
		n = 0
		data_list = []
		while n < data_array.shape[0]:
			if float(data_array[n, intensity_pos]) > eps:
				data_list.append(data_array[n])
			n = n + 1

		is_rain_array = np.array(data_list).reshape([-1, num_col])

		return is_rain_array

	def event_splitter(self, index_pos, intensity_pos, dry_time_interval=30, min_vol=0.1):
		"""
		Args:
			input_npy: input np.ndarray
			index_pos: in which columns the index is
			intensity_pos: in which columns the intensity is
			dry_time_interval: the time interval between two rain events; default=15min
			min_vol: minimum volume of rainfall event; default= 0.1mm

		"""
		input_npy = np.load(self.path + self.file_name, allow_pickle=True)
		i, rain_start = 0, 0  # rain_start: time point in which raining starts
		events_list = []
		time_points = input_npy[:, index_pos]
		time_interval = np.diff(time_points)
		rain_pauses = 0
		rain_new = 0

		while i < time_interval.shape[0]:
			if time_interval[i] < dry_time_interval:
				pass
			elif dry_time_interval <= time_interval[i] < 4*dry_time_interval:
				rain_pauses = rain_pauses + 1
				if np.sum(input_npy[rain_new:i+1, intensity_pos].astype(np.float))*0.5/60 <= min_vol:

					if rain_pauses > 1:
						rain_stop = i
						events_list.append(input_npy[rain_start:rain_stop+1])
					# what if the dry time interval between drizzle and heavy rain is far larger than 15 min

					else:
						pass

					rain_start = i + 1
					rain_new = i + 1
					rain_pauses = 0


				else:

					if rain_pauses > 1:
						events_list.append(input_npy[rain_start:rain_stop+1])
						# events_list.append(input_npy[rain_new:i])
						rain_start = rain_new
						# rain_stop =i
						# rain_new = i + 1
						rain_pauses = 1

					# else:
					# 	rain_stop = i
					# 	rain_new = i + 1
					rain_stop = i
					rain_new = i + 1

			else:
				if np.sum(input_npy[rain_new:i, intensity_pos].astype(np.float))*0.5/60 > min_vol:
					if rain_pauses !=0:
						events_list.append(input_npy[rain_start:rain_stop])
						events_list.append(input_npy[rain_new:i])
					else:
						events_list.append(input_npy[rain_new:i])
				else:
					if rain_pauses !=0:
						rain_stop = i
						events_list.append(input_npy[rain_start:rain_stop])

				rain_start = i +1
				rain_new = i + 1
				rain_pauses = 0

			i = i + 1
		# --------------------------------------------------------------------------------------------------------------#
		# while i < time_interval.shape[0]:
		# 	if time_interval[i] < dry_time_interval:
		# 		pass
		# 	else:
		# 		rain_pauses = rain_pauses + 1
		# 		if np.sum(input_npy[rain_new:i+1, intensity_pos].astype(np.float)) * 0.5 / 60 <= min_vol:
		#
		# 			if rain_pauses > 1:
		# 				rain_stop = i
		# 				events_list.append(input_npy[rain_start:rain_stop+1])
		# 				# what if the dry time interval between drizzle and heavy rain is far larger than 15 min
		#
		# 			else:
		# 				pass
		#
		# 			rain_start = i + 1
		# 			rain_new = i + 1
		# 			rain_pauses = 0
		#
		# 		else:
		#
		# 			if rain_pauses > 1:
		# 				events_list.append(input_npy[rain_start:rain_stop+1])
		# 				# events_list.append(input_npy[rain_new:i])
		# 				rain_start = rain_new
		# 				# rain_stop =i
		# 				# rain_new = i + 1
		# 				rain_pauses = 1
		# 			# else:
		# 			# 	rain_stop = i
		# 			# 	rain_new = i + 1
		# 			rain_stop = i
		# 			rain_new = i + 1
		#
		#
		# 	i = i + 1
		# --------------------------------------------------------------------------------------------------------------#
		# while i < time_interval.shape[0]:
		# 	if time_interval[i] < dry_time_interval:
		# 		pass
		# 	else:
		# 		rain_pauses = rain_pauses + 1
		# 		if np.sum(input_npy[rain_new:i, intensity_pos].astype(np.float)) * 0.5 / 60 <= min_vol:
		#
		# 			if rain_pauses > 1:
		# 				rain_stop = i
		# 				events_list.append(input_npy[rain_start:rain_stop])
		# 				# what if the dry time interval between dizzel and heavy rain is far larger than 15 min
		#
		# 			else:
		# 				pass
		#
		# 			rain_start = rain_new = i + 1
		# 			rain_pauses = 0
		#
		# 		else:
		#
		# 			rain_new = i + 1
		#
		# 	i = i + 1

		return events_list

		# while i < time_interval.shape[0]:
		# 	if time_interval[i] < dry_time_interval:
		# 		pass
		# 	else:
		# 		rain_stop = i
		# 		if np.sum(input_npy[rain_start:rain_stop, intensity_pos].astype(np.float))*0.5/60 > min_vol:
		# 			events_list.append(input_npy[rain_start:rain_stop])
		# 		rain_start = i + 1
		# 	i = i + 1
		#
		# return events_list


class DataProcess(object):
	'''
	Args:
		splitted_data: the output from EventSplitter().raintime_extractor()
		path: the folder of .npy file
		file_name: .npy file name
		self.velocity, self.velocity_width, self.diameter, self.diameter_width are the metadata of the classes of drop size and velocity
		v_correction: whether use a filter to recognize the natural rain drop.
		use_splitted_data: whether use splitted_data
	'''
	def __init__(self, splitted_data, path=None, file_name=None, v_correction=True, use_splitted_data=True):
		super(DataProcess, self).__init__()
		self.velocity = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,
								  1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,
								  12,13.6,15.2,17.6,20.8))
		self.velocity_width = np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,
										0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,
										1.6,1.6,1.6,1.6,3.2,3.2))
		self.diameter = np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,
								  1.375,1.625,1.875,2.125,2.375,2.750,3.250,3.750,4.250,4.750,
								  5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5])
		self.diameter_width = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,
									 0.250,0.250,0.250,0.250,0.250,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,
									 2,2,2,2,2,3,3])
		self.v_correction = v_correction
		self.use_splitted_data = use_splitted_data
		self.splitted_data = splitted_data
		self.path = path
		self.file_name = file_name

	def v_d_Lhermitte_1988(self, d):
		'''Theoretical velocity for raindrop of specific size'''
		d = d * 10 ** (-3)
		v = 9.25 - 9.25 * np.exp(-(68000 * d ** 2 + 488 * d))

		return v

	def v_d_filter(self):

		if self.v_correction:
			filter_velocity = np.ones((32, 32))
			velocity_theory = self.v_d_Lhermitte_1988(self.diameter)
			for i in range(32):  # Class of drop size
				for j in range(32):  # Class of velocity
					if np.abs(self.velocity[j]-velocity_theory[i]) <= 0.6*velocity_theory[i]:
						filter_velocity[i, j] = 1
					else:
						filter_velocity[i, j] = 0

			return filter_velocity

	def data_extractor(self, start_time=1, chosen_time_steps=None, use_all_data=True):
		"""
		start_time: entry for the ith data, start_time >= 1

		the .npy data is a np.2darray of shape(time_points, data) ---> 2D matrix
		----------------------------------
		|	array[[..., ..., ..., ...],  |
		|	[..., ..., ..., ...],        |
		|	.                            |
		|	.                            |
		|	.                            |
		|	[..., ..., ..., ...]]        |
		----------------------------------

		If the data includes index, temperature, rainfall intensity, and raw matrix of dsd for different classes of size and velocity,
		then the shape of the data would be [time+points, 1028] for 32X32 classes of size and velocity respectively.

		"""

		# Extracting the data from the data
		if self.use_splitted_data:
			init_data = self.splitted_data
		else:
			init_data = np.load(self.path + self.file_name, allow_pickle=True)
		# print('-' * 60 + '\nThe shape of the initial data is {}'.format(init_data.shape))

		if start_time == 0:
			raise ValueError('the start_time_point should start from one !')

		if use_all_data:
			time_steps = init_data.shape[0]
			N_Di_init = init_data[:, 5]  # number of rain drops directly for disdrometer
		else:
			time_steps = chosen_time_steps
			N_Di_init = init_data[start_time-1, chosen_time_steps]

		index_init_matrix = start_time  # to choose certain time point of data;
		t = 0  # use < for the while loop, t is to count the time steps, how many time points you want to conclude A.K.A.;
		missing_data = 0

		S_eff = 10 ** (-6) * 180 * (30 - self.diameter / 2)
		v_theo = self.v_d_Lhermitte_1988(self.diameter)

		idx = np.zeros(time_steps).astype(np.int)
		date_time = np.zeros(time_steps).astype(np.object)
		tempore = np.zeros(time_steps).astype(np.int)
		I_init = np.zeros(time_steps)
		I_t_Jaffrain = np.zeros(time_steps)
		I_t = np.zeros(time_steps)
		N_Di = np.zeros((time_steps, 32))
		rho = np.zeros(time_steps)
		Nt = np.zeros(time_steps) # N_t is the sum of all drops over the range of sampled diameters
		D_3rd_m = np.zeros(time_steps) # the third moment of the DSD
		D_4th_m = np.zeros(time_steps) # the fourth moment of the DSD
		raw_matrix = np.zeros((time_steps, 32, 32))
		ND_raw_matrix= np.zeros((32, 32))

		while t < time_steps:
			# Computing the data for the Parsivel
			try:
				idx[t] = init_data[index_init_matrix-1, 0]
				date_time[t] = init_data[index_init_matrix-1, 1]
				tempore[t] = int(init_data[index_init_matrix-1, 4])
				I_init[t] = float(init_data[index_init_matrix-1, 2])
				raw_matrix[t] = init_data[index_init_matrix-1, 6].astype(np.int).reshape(32,32).T*self.v_d_filter()
				ND_raw_matrix += raw_matrix[t]

				for i in range(2, 28):  # Loop on the sizes
					for j in range(2, 32):  # Loop on the velocity
						# Auguste Gires
						N_Di[t, i] += raw_matrix[t, i, j] / (S_eff[i] * self.diameter_width[i] * self.velocity[j] * 30)
						I_t[t] += math.pi * 120 * raw_matrix[t, i, j] * self.diameter[i] ** 3 / (6 * S_eff[i] * 10 ** 6)

				for i in range(2, 28): # loop on the size
					I_t_Jaffrain[t] += 6 * math.pi * 10 ** (-4) * N_Di[t, i] * v_theo[i] * self.diameter[i] ** (3) * self.diameter_width[i]
					rho[t] = math.pi * N_Di[t, i] * self.diameter[i] ** 3 * self.diameter_width[i] / (6 * 10 ** 3)  # in g/m3
					Nt[t] += N_Di[t, i] * self.diameter_width[i]
					D_3rd_m[t] += N_Di[t, i] * (self.diameter[i]**3) * self.diameter_width[i]
					D_4th_m[t] += N_Di[t, i] * (self.diameter[i]**4) * self.diameter_width[i]

			except:
				missing_data = missing_data + 1
				for i in range(32):  # Loop on the sizes
					N_Di[t, i] = np.nan
				I_t[t] = np.nan
				I_t_Jaffrain[t] = np.nan
				rho[t] = np.nan
				Nt[t] = np.nan

			t = t + 1
			index_init_matrix = index_init_matrix + 1

		all_data = np.concatenate([idx.reshape((time_steps, -1)), date_time.reshape((time_steps, -1)),
								   I_init.reshape((time_steps, -1)), I_t.reshape((time_steps, -1)), I_t_Jaffrain.reshape((time_steps, -1)),
								   N_Di], axis=1)

		N_t = np.sum(N_Di)
		N_D = np.sum(N_Di, axis=0)
		D_m = np.sum(D_4th_m, axis=0) / np.sum(D_3rd_m, axis=0)
		f_D = N_D / N_t

		return N_Di_init, N_Di, ND_raw_matrix, all_data, N_D, N_t, D_m, f_D


class DataPlot(object):
	'''
	Args:
		splitted_data: the output from EventSplitter().raintime_extractor()
		path: the folder of .npy file
		file_name: .npy file name
		self.velocity, self.velocity_width, self.diameter, self.diameter_width are the metadata of the classes of drop size and velocity
		v_correction: whether use a filter to recognize the natural rain drop.
		use_splitted_data: whether use splitted_data
	'''
	def __init__(self, splitted_data=None, path=None, file_name=None, v_correction=True, use_splitted_data=False):
		super(DataPlot, self).__init__()
		self.velocity = np.array((0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.1,1.3,
								  1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,
								  12,13.6,15.2,17.6,20.8))
		self.velocity_width = np.array((0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,
										0.2,0.2,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,1.6,
										1.6,1.6,1.6,1.6,3.2,3.2))
		self.diameter = np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,
								  1.375,1.625,1.875,2.125,2.375,2.750,3.250,3.750,4.250,4.750,
								  5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5])
		self.diameter_width = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,
									 0.250,0.250,0.250,0.250,0.250,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,
									 2,2,2,2,2,3,3])
		self.v_correction = v_correction
		self.use_splitted_data = use_splitted_data
		self.splitted_data = splitted_data
		self.path = path
		self.file_name = file_name

	def v_d_Lhermitte_1988(self, d):
		'''Theoretical velocity for raindrop of specific size'''
		d = d * 10 ** (-3)
		v = 9.25 - 9.25 * np.exp(-(68000 * d ** 2 + 488 * d))

		return v

	def v_d_filter(self):

		if self.v_correction:
			filter_velocity = np.ones((32, 32))
			velocity_theory = self.v_d_Lhermitte_1988(self.diameter)
			for i in range(32):  # Class of drop size
				for j in range(32):  # Class of velocity
					if np.abs(self.velocity[j]-velocity_theory[i]) <= 0.6*velocity_theory[i]:
						filter_velocity[i, j] = 1
					else:
						filter_velocity[i, j] = 0

			return filter_velocity

	def temporal_intensity(self, inputs, idx_init, idx_jaff, idx_dsd, idx_date):

		if self.use_splitted_data:
			inputs = self.splitted_data

		times = np.array(range(inputs.shape[0]))*30/3600  # in hour
		plt.figure()
		plt.plot(times, inputs[:,idx_init], color='r', label='via disdrometer')
		plt.plot(times, inputs[:,idx_jaff], color='g', label='via Jaffrain')
		plt.plot(times, inputs[:,idx_dsd], color='b', label='via DSD export')
		plt.title('Event started at {}'.format(inputs[0, idx_date]), fontsize=20, color='k')
		plt.xlabel('Time (h)', fontsize=20, color='k')
		plt.ylabel('R (mm/h)', fontsize=20, color='k')
		plt.legend()
		plt.savefig(self.path + 'It_' + inputs[0, idx_date] + '.png', dpi=300, bbox_inches='tight', format='png')


	def temporal_nd(self, inputs, idx_dsd, idx_date):

		if self.use_splitted_data:
			inputs = self.splitted_data

		times = np.array(range(inputs.shape[0]))*30/3600
		time_steps = inputs.shape[0]
		size_classes = 32
		list_colors = []
		jet = plt.get_cmap('jet')
		cNorm = colors.Normalize(vmin=0, vmax=size_classes - 1)
		scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

		for i in range(0, size_classes):
			list_colors.append(scalarMap.to_rgba(i))
		# print(colorVal)

		max_value = np.log10(np.nanmax(inputs[:, idx_dsd:idx_dsd+size_classes].astype(np.float)))

		plt.clf()

		ax = plt.axes([1, 1, 1, 1])  # fig.add_subplot(312)
		try:
			codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
			for t in range(time_steps - 1):
				for i in range(26):

					if inputs[t, i + idx_dsd] > 0:
						verts = [(times[t], self.diameter[i]), (times[t], self.diameter[i + 1]), (times[t + 1],
								self.diameter[i + 1]), (times[t + 1], self.diameter[i]), (times[t], self.diameter[i])]
						ind_color = np.floor(32 * (np.log10(inputs[t, i + idx_dsd] - 0.0001)) / max_value)
						ind_color = ind_color.astype(int)

						if ind_color == -1:
							ind_color = 0

						path = Path(verts, codes)
						patch = patches.PathPatch(path, facecolor=list_colors[ind_color], lw=0)
						ax.add_patch(patch)
		except:
			print('Error')

		plt.xlabel(r'$Time\/(h)$', fontsize=20, color='k')
		plt.ylabel(r'$\mathit{D}\/\/(mm)$', fontsize=20, color='k')
		plt.title('Event started at {}'.format(inputs[0, idx_date]), fontsize=20, color='k')
		plt.xlim(0, times[-1])
		for xtick in ax.get_xticklabels():
			plt.setp(xtick, fontsize=12)
		for ytick in ax.get_yticklabels():
			plt.setp(ytick, fontsize=12)
		ax.set_xlim(0, times[-1])
		ax.set_ylim(0, 9)

		ax_colorbar = plt.axes([2.1, 1, 0.05, 0.83])
		cmap = colors.ListedColormap(list_colors)
		# cmap.set_over((1., 0., 0.))
		# cmap.set_under((0., 0., 1.))
		try:
			bounds = range(33) * max_value / 32  # [-1., -.5, 0., .5, 1.]
			norm = colors.BoundaryNorm(bounds, cmap.N)
			# cax, kw = colorbar.make_axes(ax)
			cb3 = colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, boundaries=bounds, extendfrac='auto',
										ticks=bounds[0:-1:2], spacing='uniform', orientation='vertical')
			cb3.set_label(r'$log_{10}\/\mathit{N(D)}$', fontsize=20)
			cb3.set_ticklabels(np.floor(bounds[0:-1:2] * 100) / 100)
		except:
			print('Error')

		plt.savefig(self.path + 'NDt_' + inputs[0, idx_date] + '.png', dpi=300, bbox_inches='tight', format='png')
		# plt.close()


	def size_velocity(self, rawmatrix_inputs, info_inputs, idx_date):
		# inputs must be raw_matrix after filter

		size_classes = 32
		list_colors = []
		jet = plt.get_cmap('jet')
		cNorm = colors.Normalize(vmin=0, vmax=size_classes - 1)
		scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

		inputs = rawmatrix_inputs

		for i in range(0, size_classes):
			list_colors.append(scalarMap.to_rgba(i))
		plt.clf()
		log10_num_all_drops = np.zeros((32, 32))
		for i in range(2, 28):
			for j in range(2, 32):
				log10_num_all_drops[i, j] = np.log10(inputs[i, j] / (self.diameter_width[i] * self.velocity_width[j]))

		max_value = np.max(log10_num_all_drops)

		codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

		ax = plt.axes([1, 1, 1, 1])
		codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
		for i in range(26):  # A rojouter après
			for j in range(30):
				if log10_num_all_drops[i, j] > 0:
					verts = [(self.diameter[i], self.velocity[j]), (self.diameter[i], self.velocity[j + 1]),
							 (self.diameter[i + 1], self.velocity[j + 1]),(self.diameter[i + 1], self.velocity[j]),
							 (self.diameter[i], self.diameter[j])]
					ind_color = np.floor(32 * (log10_num_all_drops[i, j] - 0.00001) / max_value)
					ind_color = ind_color.astype(int)
					path = Path(verts, codes)
					patch = patches.PathPatch(path, facecolor=list_colors[ind_color], lw=0)
					ax.add_patch(patch)
		plt.plot(self.diameter, self.v_d_Lhermitte_1988(self.diameter), color='k')
		plt.xlabel(r'$\mathit{D}\/\/(mm)$', fontsize=14, color='k')
		plt.ylabel(r'$\mathit{V}\/\/(m.s^{-1})$',fontsize=20,color='k')
		plt.title('Event started at {}'.format(info_inputs[0, idx_date]), fontsize=14, color='k')
		for xtick in ax.get_xticklabels():
			plt.setp(xtick, fontsize=14)
		for ytick in ax.get_yticklabels():
			plt.setp(ytick, fontsize=14)
		ax.set_xlim(0, 9)
		ax.set_ylim(0, 12)

		ax_colorbar = plt.axes([2.1, 1, 0.05, 0.83])
		cmap = colors.ListedColormap(list_colors)
		try:
			bounds = range(33) * max_value / 32  # [-1., -.5, 0., .5, 1.]
			norm = colors.BoundaryNorm(bounds, cmap.N)
			#    cax, kw = colorbar.make_axes(ax)
			cb3 = colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, boundaries=bounds, extendfrac='auto',
										ticks=bounds[0:-1:3], spacing='uniform', orientation='vertical')
			cb3.set_ticklabels(np.floor(bounds[0:-1:3] * 100) / 100)
			cb3.set_label('log10(all_drops)')
		except:
			print('Error')

		plt.savefig(self.path + 'svmap_' + info_inputs[0, idx_date] + '.png', dpi=300, bbox_inches='tight', format='png')
		# plt.close()

	def num_size(self, inputs, info_inputs, inputs1, inputs2, idx_date):

		plt.figure(6)
		plt.clf()
		# p1, = plt.plot(D_PWS[0:29], np.nanmean(N_D_emp_PWS, axis=0)[0:29] * D_PWS[0:29] ** 3, color='r', lw=2)
		# p2, = plt.plot(D_Pars1[0:26], np.nanmean(N_D_emp_Pars1, axis=0)[0:26] * D_Pars1[0:26] ** 3, color='b', lw=2)
		p1, = plt.plot(self.diameter[0:26], np.nanmean(inputs, axis=0)[0:26] * self.diameter[0:26] ** 3, color='r', lw=2)
		p2, = plt.plot(self.diameter[0:26], np.nanmean(inputs1, axis=0)[0:26] * self.diameter[0:26] ** 3, color='g', lw=2)
		p3, = plt.plot(self.diameter[0:26], np.nanmean(inputs2, axis=0)[0:26] * self.diameter[0:26] ** 3, color='b', lw=2)
		plt.legend([p1, p2, p3], ['event1', 'event2', 'event3'], loc='upper right', frameon=False)
		# plt.legend(p3, 'station40', loc='upper right', frameon=False)
		plt.xlabel(r'$\mathit{D}\/\/(mm)$', fontsize=20, color='k')
		plt.ylabel(r'$\mathit{N(D)}\/\mathit{D}^3$', fontsize=20, color='k')
		plt.title('', fontsize=20, color='k')
		ax = plt.gca()
		for xtick in ax.get_xticklabels():
			plt.setp(xtick, fontsize=14)
		for ytick in ax.get_yticklabels():
			plt.setp(ytick, fontsize=14)
		ax.set_xlim(0, 9)
		plt.savefig(self.path + 'ns_' + info_inputs[0, idx_date] + '.png', dpi=300, bbox_inches='tight',format='png')
		# plt.close()


if __name__ == '__main__':
	path = '/home/xiaochenzheng/Desktop/Master_Project/2019_Disdros_TBA/Data/'
	file_name = 'station40.npy'
	eventsplitter = EventSplitter(path, file_name)
	eventsss = eventsplitter.event_splitter(0, 2)
	dataextractor = DataProcess(eventsss[5])
	a, b, c, d = dataextractor.data_extractor()
	dataextractor = DataProcess(eventsss[6])
	a1, b1, c1, d1 = dataextractor.data_extractor()
	dataextractor = DataProcess(eventsss[7])
	a2, b2, c2, d2 = dataextractor.data_extractor()

	dataplot = DataPlot(path=path)
	dataplot.temporal_nd(d, 5, 1)
	dataplot.temporal_intensity(d, 2, 3, 4, 1)
	dataplot.size_velocity(c, d, 1)
	dataplot.num_size(b, d, b1, b2, 1)