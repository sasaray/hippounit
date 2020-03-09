import sciunit
from sciunit import Capability

class BlockNaChannels(sciunit.Capability):
	"""Indicates that the model receives one or multiple synapses"""

	def block_Na(self):
		""" Must return numpy arrays containing the time and voltage values (at the soma and at the synaptic location )"""
		raise NotImplementedError()

	def block_Na_channels(self):

		self.block_Na()
