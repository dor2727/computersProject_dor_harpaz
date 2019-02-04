import os
import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def avg(z, dy):
	return (
		sum([
			zi / (dyi**2)
			for zi, dyi in zip(z, dy)
		])
		 /
		sum([
			1 / (dyi**2)
			for dyi in dy
		])
	)

class DataReader(object):
	def __init__(self, filename):
		self.filename = filename

		self.file_handle = open(filename, 'r')

	def __del__(self):
		self.file_handle.close()

	def read(self):
		self._raw_data = list(csv.reader(self.file_handle, delimiter=' '))

		# get all rows containing data (or the headers) until the seperator row in reached
		self._data = []
		for row in self._raw_data:
			if not row:
				break
			self._data.append(row)

		self._discover_mode()
		self._validate_data_length()
		self._convert_to_rows()
		self._initialize_dict()
		self.n = len(self.data_dict['x'])
		self._validate_data_value()

	def _discover_mode(self):
		if self._data[0][0].lower() != 'x':
			print("[!] invalid data - no 'x' column found")
			sys.exit(1) # 1 is error code

		if self._data[0][1].lower() == 'dx':
			self.mode = 'rows'
		elif self._data[1][0].lower() == 'dx':
			self.mode = 'columns'
		else:
			print("[!] invalid data - no 'x' column found")
			sys.exit(1) # 1 is error code

	def _validate_data_length(self):
		if self.mode == 'rows':
			for row in self._data:
				# the empty row - a row seperator
				if not row:
					break
				# all(rows) will verify that every item is non-empty, i.e. no empty string
				if len(row) != 4 or not all(row):
					print("Input file error: Data lists are not the same length.")
					sys.exit(1) # 1 is error code
		elif self.mode == 'columns':
			length_of_row = len(self._data[0])
			for row in self._data:
				# the empty row - a row seperator
				if not row:
					break
				if len(row) != length_of_row or not all(row):
					print("Input file error: Data lists are not the same length.")
					sys.exit(1) # 1 is error code

	def _convert_to_rows(self):
		if self.mode == 'rows':
			return
		new_data = []
		length_of_row = len(self._data[0])
		for i in range(length_of_row):

			new_data.append([j[i] for j in self._data])

		self._data = new_data
		self.mode = 'rows'

	def _initialize_dict(self):
		"""
		dictionary comprehension
		automatically get the key name from the headers.
		    allowing for more variations in the input data
		sets a list containing all the data for every specific category
		    e.g. 'x', 'dx' etc.
		runs on a loop for all input headers.
		"""
		try:
			self.data_dict = {
				self._data[0][i].lower():
				np.array([float(row[i]) for row in self._data[1:]])
				for i in range(len(self._data[0]))
			}
		except:
			print("[!] float convertion failed - invalid input data")
			sys.exit(1) # 1 is error code

	def _validate_data_value(self):
		"""
		validating that all the value in dx and dy a are positive
		"""
		if not all(
			map(
				lambda value: value > 0,
				[ *self.data_dict['dx'], *self.data_dict['dy'] ]
			)
		):
			print("Input file error: Not all uncertainties are positive.")
			sys.exit(1) # 1 is error code
		return True

	def fit(self):
		_avg = lambda z: avg(z, self.data_dict['dy'])

		a = (
			(
				_avg(self.data_dict['x'] * self.data_dict['y'])
				 -
				(
					_avg(self.data_dict['x'])
					*
					_avg(self.data_dict['y'])
				)
			) / (
				_avg(self.data_dict['x'] ** 2)
				 -
				_avg(self.data_dict['x'])  ** 2
			)
		)
		da = (
			_avg(self.data_dict['dy'] ** 2)
			 /
			self.n * (
				_avg(self.data_dict['x'] ** 2)
				 -
				_avg(self.data_dict['x'])  ** 2
			)
		)
		b = (
			_avg(self.data_dict['y'])
			 -
			a * _avg(self.data_dict['x'])
		)
		db = (
			(
				_avg(self.data_dict['dy'] ** 2)
				 *
				_avg(self.data_dict['x'] ** 2)
			)
			 /
			(
				self.n
				 *
				(
					_avg(self.data_dict['x'] ** 2)
					 -
					_avg(self.data_dict['x'])  ** 2
				)
			)
		)
		return a, da, b, db

	def calculate_chi_square(self, a, b):
		chi = sum([
			(
				(
					self.data_dict['y'][i]
					 -
					(a * self.data_dict['x'][i] + b)
				) / (
					self.data_dict['dy'][i]
				)
			) ** 2
			for i in range(self.n)
		])
		chi_reduced = chi / (self.n - 2)

		return chi, chi_reduced

	def extract_plot_labels(self):
		"""
		the last 2 rows has the format of:
			<axis name - 1 letter> axis: <axis title>
		thus, self._raw_data[row_index][0] will be the axis,
			  self._raw_data[row_index][1] will be "axis:"
			  and the rest will be the requested title
		"""
		axis_label_index = self._raw_data.index([]) # an empty list - a row seperator
		labels = {}
		labels[self._raw_data[axis_label_index+1][0].lower()] = \
			' '.join(self._raw_data[axis_label_index+1][2:])
		labels[self._raw_data[axis_label_index+2][0].lower()] = \
			' '.join(self._raw_data[axis_label_index+2][2:])
		return labels

def fit_linear(filename):
	if not os.path.exists(filename):
		print("[!] file not found!")
		sys.exit(1) # 1 is error code

	# read the data
	data_reader = DataReader(filename)
	data_reader.read()
	x  = data_reader.data_dict['x']
	dx = data_reader.data_dict['dx']
	y  = data_reader.data_dict['y']
	dy = data_reader.data_dict['dy']

	# fit
	a, da, b, db = data_reader.fit()

	chi, chi_reduced = data_reader.calculate_chi_square(a, b)

	# print results
	print("a = %s +- %s" % (a, da))
	print("b = %s +- %s" % (b, db))
	print("chi2 = %s" % chi)
	print("chi2_reduced = %s" % chi_reduced)
	
	# plot error-crosses
	plt.errorbar(x, y, yerr=dy, xerr=dx, fmt='b+')

	# plot fit
	fy = [a*i+b for i in x]
	plt.plot(x, fy, 'r')

	# set axis labels
	labels = data_reader.extract_plot_labels()
	plt.xlabel(labels['x'])
	plt.ylabel(labels['y'])

	plt.show()

	export_filename = os.path.join(
		os.path.dirname(filename),
		"linear_fit.svg"
	)
	plt.savefig(export_filename)

class Bonus(DataReader):
	def _extract_fit_parameters(self):
		# get last empty row - line seperator
		parameter_index = self._raw_data[::-1].index([])
		parameter_index *= -1
		self.parameters = {}
		for i in (0,1):
			parameter_name = self._raw_data[parameter_index+i][0].lower()
			parameter_range = list(map(float, self._raw_data[parameter_index][1:]))
			parameter_range = [
				float(self._raw_data[parameter_index+i][1]),
				float(self._raw_data[parameter_index+i][2]),
			]
			parameter_step = float(self._raw_data[parameter_index+i][3])

			if parameter_step < 0:
				parameter_step *= (-1)
			elif parameter_step == 0:
				print("[!] invalid step!")
				sys.exit(1)

			self.parameters[parameter_name] = {
				"start": min(parameter_range),
				"stop" : max(parameter_range),
				"step" : parameter_step,
			}

	# override
	def read(self):
		super().read()
		self._extract_fit_parameters()

	# override
	def fit(self):
		best_a = None
		best_b = None
		best_chi = float('inf')
		best_chi_reduced = float('inf')
		data_for_graph = []

		a = self.parameters['a']["start"]
		while a <= self.parameters['a']["stop"]:
			b = self.parameters['b']["start"]
			while b <= self.parameters['b']["stop"]:

				chi, chi_reduced = self.calculate_chi_square(a, b)
				data_for_graph.append([a, b, chi, chi_reduced])

				if chi < best_chi:
					best_a = a
					best_b = b
					best_chi = chi
					best_chi_reduced = chi_reduced

				b += self.parameters['b']["step"]
			a += self.parameters['a']["step"]

		return {
			'a': best_a,
			'b': best_b,
			'chi': best_chi,
			'chi_reduced': best_chi_reduced,
			'plot_data': data_for_graph,
		}

def search_best_parameter(filename):
	if not os.path.exists(filename):
		print("[!] file not found!")
		sys.exit(1) # 1 is error code

	# read the data
	data_reader = Bonus(filename)
	data_reader.read()
	x  = data_reader.data_dict['x']
	dx = data_reader.data_dict['dx']
	y  = data_reader.data_dict['y']
	dy = data_reader.data_dict['dy']

	# fit
	fit_parameters = data_reader.fit()
	# for convenince
	a = fit_parameters['a']
	b = fit_parameters['b']

	# print results
	print("a = %s +- %s" % (a, data_reader.parameters['a']["step"]))
	print("b = %s +- %s" % (b, data_reader.parameters['b']["step"]))
	print("chi2 = %s" % fit_parameters['chi'])
	print("chi2_reduced = %s" % fit_parameters['chi_reduced'])
	
	# plot error-crosses
	plt.errorbar(x, y, yerr=dy, xerr=dx, fmt='b+')

	# plot fit
	fy = [a*i+b for i in x]
	plt.plot(x, fy, 'r')

	# set axis labels
	labels = data_reader.extract_plot_labels()
	plt.xlabel(labels['x'])
	plt.ylabel(labels['y'])

	plt.show()

	export_filename = os.path.join(
		os.path.dirname(filename),
		"linear_fit.svg"
	)
	plt.savefig(export_filename)

	#####################
	### plot chi of a ###
	#####################

	# filter only the a's assosiated with the best b
	plot_data = list(filter(
		lambda value: value[1] == b,
		fit_parameters["plot_data"]
	))
	plot_data_a   = [row[0] for row in plot_data]
	plot_data_chi = [row[2] for row in plot_data]

	# blue line is the default
	plt.plot(plot_data_a, plot_data_chi)

	plt.xlabel('a')
	plt.ylabel('chi2(a, b = %.2f)' % b)

	plt.show()

	export_filename = os.path.join(
		os.path.dirname(filename),
		"numeric_sampling.svg"
	)
	plt.savefig(export_filename)

filename = '/home/me/Dropbox/Courses/University/year_1/semester_1/Computers/Project/inputOutputExamples/workingCols/input.txt'
filename = '/home/me/Dropbox/Courses/University/year_1/semester_1/Computers/Project/inputOutputExamples/errSigma/input.txt'
filename = '/home/me/Dropbox/Courses/University/year_1/semester_1/Computers/Project/inputOutputExamples/errDataLength/input.txt'
filename = '/home/me/Dropbox/Courses/University/year_1/semester_1/Computers/Project/inputOutputExamples/workingRows/input.txt'
bonus = '/home/me/Dropbox/Courses/University/year_1/semester_1/Computers/Project/inputOutputExamples/bonus/input.txt'