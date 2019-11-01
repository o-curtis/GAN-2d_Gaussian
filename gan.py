#!/usr/bin/env python

# Generative Adversarial Networks (GAN) example in PyTorch. Tested with PyTorch 0.4.1, Python 3.6.7 (Nov 2018)
# Original 1D code by Dev Nag
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os

matplotlib_is_available = True
try:
  import matplotlib.pylab as plt
except ImportError:
  print("Will skip plotting; matplotlib is not available.")
  matplotlib_is_available = False

# Data params
data_mean = 4
data_stddev = 1.25

# ### Uncomment only one of these to define what data is actually sent to the Discriminator
#(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
#(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)
#(name, preprocess, d_input_func) = ("Data and diffs", lambda data: decorate_with_diffs(data, 1.0), lambda x: x * 2)
(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 8)

print("Using data [%s]" % (name))

# ##### DATA: Target data and generator input data

def get_distribution_sampler(mu, sigma):
	return lambda n: torch.Tensor(np.random.normal(mu, sigma, (2, n)))  # Gaussian

def get_generator_input_sampler():
	return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, f):
		super(Generator, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_size)
		self.map2 = nn.Linear(hidden_size, hidden_size)
		self.map3 = nn.Linear(hidden_size, output_size)
		self.f = f

	def forward(self, x):
		x = self.map1(x)
		x = self.f(x)
		x = self.map2(x)
		x = self.f(x)
		x = self.map3(x)
		return x

class Discriminator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, f):
		super(Discriminator, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_size)
		self.map2 = nn.Linear(hidden_size, hidden_size)
		self.map3 = nn.Linear(hidden_size, output_size)
		self.f = f

	def forward(self, x):
		x = self.f(self.map1(x))
		x = self.f(self.map2(x))
		return self.f(self.map3(x))

def extract(v):
	return v.data.storage().tolist()

def stats(d):
	return [np.mean(d), np.std(d)]
	
def make_movie(list):
	os.system("mkdir movie_plots")
	for k in range(len(list)):
		if k < 10:
			file_name = "img-000"+str(k)+".png"
		elif k < 100:
			file_name = "img-00" +str(k)+".png"
		elif k < 1000:
			file_name = "img-0" + str(k) +".png"
		else:
			file_name = "img-"+str(k) + ".png"
		plt.hist2d(list[k][0],list[k][1], bins=50)
		plt.title("Histogram of Generated Distribution")
		plt.grid(True)
		plt.savefig(file_name)
		os.system("mv "+ file_name + " movie_plots")
		plt.clf()

def get_moments(d):
	# Return the first 4 moments of the data provided
	mean = torch.mean(d,1,keepdim=True)
	diffs = d - mean
	var = torch.mean(torch.pow(diffs, 2.0),1,keepdim=True)
	std = torch.pow(var, 0.5)
	zscores = diffs / std
	skews = torch.mean(torch.pow(zscores, 3.0),1,keepdim=True)
	kurtoses = torch.mean(torch.pow(zscores, 4.0),1,keepdim=True) - 3.0  # excess kurtosis, should be 0 for Gaussian
	final = torch.cat((mean.reshape(2,), std.reshape(2,), skews.reshape(2,), kurtoses.reshape(2,)))
	return final

def decorate_with_diffs(data, exponent, remove_raw_data=False):
	mean = torch.mean(data.data, 1, keepdim=True)
	mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
	diffs = torch.pow(data - Variable(mean_broadcast), exponent)
	if remove_raw_data:
		return torch.cat([diffs], 1)
	else:
		return torch.cat([data, diffs], 1)

def save_model(G,D):
	torch.save({'state_dict': G.state_dict()}, 'G.pt')
	torch.save({'state_dict': D.state_dict()}, 'D.pt')
	
def load_model(G,D):
	print("Loading previous model")
	G_state_dict = torch.load('G.pt')['state_dict']
	G.load_state_dict(G_state_dict)
	D_state_dict = torch.load('D.pt')['state_dict']
	D.load_state_dict(D_state_dict)

def train():

	generated_distributions = []
	G_losses = []
	D_fake_losses = []
	D_real_losses = []

	# Model parameters
	g_input_size = 1	  # Random noise dimension coming into generator, per output vector
	g_hidden_size = 5	 # Generator complexity
	g_output_size = 2	 # Size of generated output vector
	d_input_size = 10000	# Minibatch size - cardinality of distributions
	d_hidden_size = 10	# Discriminator complexity
	d_output_size = 1	 # Single dimension for 'real' vs. 'fake' classification
	minibatch_size = d_input_size

	d_learning_rate = 1e-3
	g_learning_rate = 1e-3
	sgd_momentum = 0.9

	num_epochs = 1000
	print_interval = 10
	d_steps = 20
	g_steps = 20

	dfe, dre, ge = 0, 0, 0
	d_real_data, d_fake_data, g_fake_data = None, None, None

	discriminator_activation_function = torch.sigmoid
	generator_activation_function = torch.tanh

	d_sampler = get_distribution_sampler(data_mean, data_stddev)
	gi_sampler = get_generator_input_sampler()
	G = Generator(input_size=g_input_size,
				  hidden_size=g_hidden_size,
				  output_size=g_output_size,
				  f=generator_activation_function)
	D = Discriminator(input_size=d_input_func(d_input_size),
					  hidden_size=d_hidden_size,
					  output_size=d_output_size,
					  f=discriminator_activation_function)
	criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
	d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
	g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)
	
	try:
		load_model(G,D)
	except:
		pass

	for epoch in range(num_epochs):
		for d_index in range(d_steps):
			# 1. Train D on real+fake
			D.zero_grad()

			#  1A: Train D on real
			d_real_data = Variable(d_sampler(d_input_size))
			d_real_decision = D(preprocess(d_real_data))
			d_real_error = criterion(d_real_decision, Variable(torch.ones([1,1])))  # ones = true
			d_real_error.backward() # compute/store gradients, but don't change params

			#  1B: Train D on fake
			d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
			d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
			d_fake_decision = D(preprocess(d_fake_data.t()))
			d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1,1])))  # zeros = fake
			d_fake_error.backward()
			d_optimizer.step()	 # Only optimizes D's parameters; changes based on stored gradients from backward()

			dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

		for g_index in range(g_steps):
			# 2. Train G on D's response (but DO NOT train D on these labels)
			G.zero_grad()

			gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
			g_fake_data = G(gen_input)
			g_fake_data.reshape(2,d_input_size)
			dg_fake_decision = D(preprocess(g_fake_data.t()))
			g_error = criterion(dg_fake_decision, Variable(torch.ones([1,1])))  # Train G to pretend it's genuine

			g_error.backward()
			g_optimizer.step()  # Only optimizes G's parameters
			ge = extract(g_error)[0]

		if epoch % print_interval == 0:
			print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
				  (epoch, dre, dfe, ge, stats(extract(d_real_data)), stats(extract(d_fake_data))))
			reformat_data = g_fake_data.reshape(2,d_input_size)
			generated_distributions.append([reformat_data[0].tolist(),reformat_data[1].tolist()])
			G_losses.append(g_error.item())
			D_fake_losses.append(d_fake_error.item())
			D_real_losses.append(d_real_error.item())
				
	make_movie(generated_distributions)
	save_model(G,D)
	
	print(G_losses[0])
	plt.plot(G_losses, label="G")
	plt.plot(D_real_losses, label="RD")
	plt.plot(D_fake_losses, label="FD")
	plt.xlabel("epoch")
	plt.ylabel("Loss")
	plt.grid(True)
	plt.legend()
	plt.show()
	plt.clf()

train()
