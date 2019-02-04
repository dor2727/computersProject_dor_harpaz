import os
import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

"""
this code was tested on:
Debian 4.9 x86_64
python: 3.5.3 (default, Sep 27 2018, 17:25:39) \n[GCC 6.3.0 20170516]
numpy: 1.16.0
matplotlib: 3.0.2
"""

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
		"""
		Read the content of the file, parse it as a csv (with a delimiter set to be space)
		Then, parse the data, and initialize inner parameters
		"""
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
		"""
		identify if the data is in rows or columns.
		note: the object in the first row and first column is expected to be 'x'
		"""
		if self._data[0][0].lower() != 'x':
			print("[!] invalid data - no 'x' column found")
			sys.exit(1) # 1 is error code

		if self._data[0][1].lower() == 'dx':
			self.mode = 'rows'
		elif self._data[1][0].lower() == 'dx':
			self.mode = 'columns'
		else:
			print("[!] invalid data - no 'dx' column found")
			sys.exit(1) # 1 is error code

	def _validate_data_length(self):
		"""
		validate that every row and column have the same length
		plus, verify that there are no empty items
		"""
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
		"""
		if the mode is columns, convert it to rows, for convinience
		"""
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
		a, da, b, db = self.fit_linear()

		chi, chi_reduced = self.calculate_chi_square(a, b)

		self.fit_parameters = {
			'a': a,
			'da': da,
			'b': b,
			'db': db,
			'chi': chi,
			'chi_reduced': chi_reduced,
		}
		return self.fit_parameters

	# this function returns (a, da, b, db)
	def fit_linear(self):
		_avg = lambda z: avg(z, self.data_dict['dy'])

		# for convenience and readability
		d = self.data_dict

		a = (
			(
				_avg(d['x'] * d['y'])
				 -
				(
					_avg(d['x'])
					*
					_avg(d['y'])
				)
			) / (
				_avg(d['x'] ** 2)
				 -
				_avg(d['x'])  ** 2
			)
		)
		da = (
			_avg(d['dy'] ** 2)
			 /
			self.n * (
				_avg(d['x'] ** 2)
				 -
				_avg(d['x'])  ** 2
			)
		)
		b = (
			_avg(d['y'])
			 -
			a * _avg(d['x'])
		)
		db = (
			(
				_avg(d['dy'] ** 2)
				 *
				_avg(d['x'] ** 2)
			)
			 /
			(
				self.n
				 *
				(
					_avg(d['x'] ** 2)
					 -
					_avg(d['x'])  ** 2
				)
			)
		)
		return a, da, b, db

	# this function returns (chi, chi_reduced)
	def calculate_chi_square(self, a, b):
		# for convenience and readability
		d = self.data_dict

		chi = sum([
			(
				(
					d['y'][i]
					 -
					(a * d['x'][i] + b)
				) / (
					d['dy'][i]
				)
			) ** 2
			for i in range(self.n)
		])
		chi_reduced = chi / (self.n - 2)

		return chi, chi_reduced

	def extract_plot_labels(self):
		"""
		the data format goes like:
			data
			data
			data
			<empty row - a row seperator>
			<axis name - 1 letter> axis: <axis title>
			<axis name - 1 letter> axis: <axis title>

		thus, self._raw_data[row_index][0] will be the axis,
			  self._raw_data[row_index][1] will be "axis:"
			  and the rest will be the requested title
		"""
		axis_label_index = self._raw_data.index([]) # an empty list - a row seperator

		labels = {}

		axis_name  = lambda index: self._raw_data[index][0].lower()
		axis_title = lambda index: ' '.join(self._raw_data[index][2:])

		labels[axis_name(axis_label_index+1)] = axis_title(axis_label_index+1)
		labels[axis_name(axis_label_index+2)] = axis_title(axis_label_index+2)

		return labels


	def plot_fit(self):
		self.fit()

		self._print_fit_results(self.fit_parameters)
		self._plot_errorbar()
		self._plot_lin_fit(self.fit_parameters)
		self._plot_set_labels()

		plt.show()

		self._save_plot("linear_fit.svg")

	def _print_fit_results(self, fit_parameters):
		# print results
		print("a = %s +- %s" % (fit_parameters['a'], fit_parameters['da']))
		print("b = %s +- %s" % (fit_parameters['b'], fit_parameters['db']))
		print("chi2 = %s"  % (fit_parameters['chi']))
		print("chi2_reduced = %s" % (fit_parameters['chi_reduced']))

	def _plot_errorbar(self):
		plt.errorbar(
			self.data_dict['x'],
			self.data_dict['y'],
			yerr=self.data_dict['dy'],
			xerr=self.data_dict['dx'],
			fmt='b+'
		)

	def _plot_lin_fit(self, fit_parameters):
		fy = [
			fit_parameters['a'] * i + fit_parameters['b']
			for i in self.data_dict['x']
		]
		red_line = 'r'
		plt.plot(self.data_dict['x'], fy, red_line) 

	def _plot_set_labels(self):
		labels = self.extract_plot_labels()
		plt.xlabel(labels['x'])
		plt.ylabel(labels['y'])

	def _save_plot(self, filename):
		plt.savefig(os.path.join(
			os.path.dirname(self.filename),
			filename
		))

def fit_linear(filename):
	if not os.path.exists(filename):
		print("[!] file not found!")
		sys.exit(1) # 1 is error code

	# read the data
	data_reader = DataReader(filename)
	data_reader.read()
	data_reader.plot_fit()

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

		self.fit_parameters = {
			'a': best_a,
			'da': self.parameters['a']["step"],
			'b': best_b,
			'db': self.parameters['b']["step"],
			'chi': best_chi,
			'chi_reduced': best_chi_reduced,
			'plot_data': data_for_graph,
		}
		return self.fit_parameters

	# override
	def extract_plot_labels(self):
		return {
			'x': 'a',
			'y': 'chi2(a, b = %.2f)' % self.fit_parameters['b'],
		}
	# override
	def plot_fit(self):
		super().plot_fit()

		self._plot_chi_of_a()

		self._plot_set_labels()
		plt.show()

		self._save_plot("numeric_sampling.svg")

	def _plot_chi_of_a(self):
		# filter only the a's assosiated with the best b
		plot_data = list(filter(
			lambda value: value[1] == self.fit_parameters["b"],
			self.fit_parameters["plot_data"]
		))
		plot_data_a   = [row[0] for row in plot_data]
		plot_data_chi = [row[2] for row in plot_data]

		# blue line is the default
		plt.plot(plot_data_a, plot_data_chi)

def search_best_parameter(filename):
	if not os.path.exists(filename):
		print("[!] file not found!")
		sys.exit(1) # 1 is error code

	# read the data
	data_reader = Bonus(filename)
	data_reader.read()
	data_reader.plot_fit()
