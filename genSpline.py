#!/usr/bin/env python


"""

	Spline-based genetic optimization class



	Author: 	Martin Laprise
			    Universite Laval
				martin.laprise.1@ulaval.ca
                 

Copyright (C) 2007-2010 Martin Laprise (mlaprise@gmail.com)

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 dated June, 1991.

This software is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANDABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA


"""

import numpy
import numpy.random
from numpy import *
import scipy
import pylab as pl
import scipy.interpolate
from scipy import signal
from scipy import stats
import time
import copy


def gaussianPulse(t, FWHM, t0, P0 = 1.0, m = 1, C = 0):
	"""
	Geneate a gaussian/supergaussiance envelope pulse

		* field_amp: 	output gaussian pulse envellope (amplitude).
		* t:     		vector of times at which to compute u
		* t0:    		center of pulse (default = 0)
		* FWHM:   		full-width at half-intensity of pulse (default = 1)
		* P0:    		peak intensity of the pulse @ t=t0 (default = 1)
		* m:     		Gaussian order (default = 1)
		* C:     		chirp parameter (default = 0)
	"""

	t_zero = FWHM/sqrt(4.0*log(2.0))
	amp = sqrt(P0)
	real_exp_arg = -pow(((t-t0)/t_zero),2.0*m)/2.0
	euler1 = cos(-C*real_exp_arg)
	euler2 = sin(-C*real_exp_arg)
	return amp*exp(real_exp_arg)*euler1 + amp*exp(real_exp_arg)*euler2*1.0j


class IndividualReal:
	'''
	Individual Class
	'''

	def __init__(self, geneLength, individualLength, func, baseVal = 0.0, varVal = 1.0):
		
		# Constructor
		self.length = 0
		self.fitness = 0.0
		self.fitnessFunc = func
		self.fitnessComputed = 0
		self.age = 0
		self.xBound = [0.0,1.0]
		self.gLength = geneLength
		self.iLength = individualLength
		self.baseVal = baseVal
		self.varVal = varVal

		self.x = pl.linspace(0.0,1.0, geneLength)
		self.y = numpy.zeros(geneLength, float)
		self.y = baseVal + numpy.random.random(geneLength)*varVal

		self.x_int = pl.linspace(self.x.min(), self.x.max(), individualLength)
		self.y_int = pl.arange(individualLength, dtype=float)
		self.birth()

	def plotGene(self):
		'''
		Plot the gene
		'''
		pl.plot(self.x, self.y, '.')
		pl.grid(True)
		pl.show()

	def plotIndividual(self):
		'''
		Plot the individual
		'''
		pl.plot(self.x_int, self.y_int)
		pl.grid(True)
		pl.show()

	def plot(self):
		'''
		Plot the individual and the gene
		'''
		pl.plot(self.x, self.y, '.')
		pl.plot(self.x_int, self.y_int)
		pl.grid(True)
		pl.show()

	def mutation(self, strength = 0.1):
		'''
		Single gene mutation
		'''
		mutStrengthReal = strength
		mutMaxSizeReal = self.gLength/2
		mutSizeReal = int(numpy.random.random_integers(1,mutMaxSizeReal))
		mutationPosReal = int(numpy.random.random_integers(0+mutSizeReal-1,self.y.shape[0]-1-mutSizeReal))
		mutationSignReal = pl.rand()
		mutationReal = pl.rand()

		if mutationSignReal > 0.5:
			for i in range(-mutSizeReal/2, mutSizeReal/2):
				self.y.real[mutationPosReal+i] = self.y.real[mutationPosReal+i] + mutStrengthReal*self.y.real[mutationPosReal+i]*mutationReal
		else:
			for i in range(-mutSizeReal/2, mutSizeReal/2):
				self.y.real[mutationPosReal+i] = self.y.real[mutationPosReal+i] - mutStrengthReal*self.y.real[mutationPosReal+i]*mutationReal

		self.birth()

	def mutations(self, nbr, strength):
		'''
		Multiple gene mutations
		'''
		for i in range(nbr):
			self.mutation(strength)
	
			
	def birth(self):
		'''
		Create the individual (compute the spline curve)
		'''
		spline = scipy.interpolate.splrep(self.x, self.y)
		self.y_int = scipy.interpolate.splev(self.x_int,spline)


	def __add__(self,Other):
		'''
		Overloading of the ADD operator for the Recombination operator
		Whole Arithmetic Recombination
		'''
		alpha = 0.5
		Child = Individual(self.gLength, self.iLength, self.fitnessFunc, self.baseVal, self.varVal)

		try:
			for i in range(self.gLength):
				Child.y[i] = alpha*self.y[i] + (1-alpha)*Other.y[i]
		except IndexError:
			print " !! Individual must the same length for proper Recombination !! "
		
		Child.birth()
		return Child

 	def __iadd__(self,Other):
		'''
		Overloading of the iADD operator for the incrementation of age
		'''
		self.age += Other
		return self

	def fitnessEval(self):
		if not self.fitnessComputed:
			try:			
				self.fitness = self.fitnessFunc(self.x_int, self.y_int)
				self.fitnessComputed = 1
			except:
				self.fitness = inf


class Individual:
	'''
	Individual Class - Complex version
	'''

	def __init__(self, geneLength, individualLength, func, baseVal = 0.0, varVal = 1.0, apodisation = True):
		
		# Constructor
		self.length = 0
		self.fitness = 0.0
		self.fitnessFunc = func
		self.fitnessComputed = 0
		self.age = 0
		self.xBound = [0.0,1.0]
		self.gLength = geneLength
		self.iLength = individualLength
		self.baseVal = baseVal
		self.varVal = varVal

		self.x = pl.linspace(0.0,1.0, geneLength)
		self.y = numpy.zeros(geneLength, complex)
		self.y.real = baseVal + numpy.random.random(geneLength)*varVal
		self.y.imag = baseVal + numpy.random.random(geneLength)*varVal
		if apodisation:
			self.y = self.y*gaussianPulse(self.x, self.x.max()/2, self.x.max()/2, 1.0, 8)

		self.x_int = pl.linspace(self.x.min(), self.x.max(), individualLength)
		self.y_int = pl.arange(individualLength, dtype=complex)
		self.birth()

	def plotGene(self):
		'''
		Plot the gene
		'''
		pl.plot(self.x, self.y, '.')
		pl.grid(True)
		pl.show()

	def plotIndividual(self):
		'''
		Plot the individual
		'''
		pl.plot(self.x_int, self.y_int)
		pl.grid(True)
		pl.show()

	def plot(self):
		'''
		Plot the individual and the gene
		'''
		pl.plot(self.x, self.y, '.')
		pl.plot(self.x_int, self.y_int)
		pl.grid(True)
		pl.show()



	def mutation(self,strength = 0.1):
		'''
		Single gene mutation - Complex version
		'''
		# Mutation du gene - real
		mutStrengthReal = strength
		mutMaxSizeReal = self.gLength/2
		mutSizeReal = int(numpy.random.random_integers(1,mutMaxSizeReal))
		mutationPosReal = int(numpy.random.random_integers(0+mutSizeReal-1,self.y.shape[0]-1-mutSizeReal))
		mutationSignReal = pl.rand()
		mutationReal = pl.rand()

		# Mutation du gene - imag
		mutStrengthImag = strength
		mutMaxSizeImag = self.gLength/2
		mutSizeImag = int(numpy.random.random_integers(1,mutMaxSizeImag))
		mutationPosImag = int(numpy.random.random_integers(0+mutSizeImag-1,self.y.shape[0]-1-mutSizeImag))
		mutationSignImag = pl.rand()
		mutationImag = pl.rand()

		if mutationSignReal > 0.5:
			for i in range(-mutSizeReal/2, mutSizeReal/2):
				self.y.real[mutationPosReal+i] = self.y.real[mutationPosReal+i] + mutStrengthReal*self.y.real[mutationPosReal+i]*mutationReal
		else:
			for i in range(-mutSizeReal/2, mutSizeReal/2):
				self.y.real[mutationPosReal+i] = self.y.real[mutationPosReal+i] - mutStrengthReal*self.y.real[mutationPosReal+i]*mutationReal

		if mutationSignImag > 0.5:
			for i in range(-mutSizeImag/2, mutSizeImag/2):
				self.y.imag[mutationPosImag+i] = self.y.imag[mutationPosImag+i] + mutStrengthImag*self.y.imag[mutationPosImag+i]*mutationImag
		else:
			for i in range(-mutSizeImag/2, mutSizeImag/2):
				self.y.imag[mutationPosImag+i] = self.y.imag[mutationPosImag+i] - mutStrengthImag*self.y.imag[mutationPosImag+i]*mutationImag

		# Compute the individual
		self.birth()


	def mutations(self, nbr, strength):
		'''
		Multiple gene mutations
		'''
		for i in range(nbr):
			self.mutation(strength)
	
			
	def birth(self):
		'''
		Create the individual (compute the spline curve)
		'''
		splineReal = scipy.interpolate.splrep(self.x, self.y.real)
		self.y_int.real = scipy.interpolate.splev(self.x_int,splineReal)
		splineImag = scipy.interpolate.splrep(self.x, self.y.imag)
		self.y_int.imag = scipy.interpolate.splev(self.x_int,splineImag)

	def __add__(self,Other):
		'''
		Overloading of the ADD operator for the Recombination operator
		Whole Arithmetic Recombination
		'''
		alpha = 0.5
		Child = Individual(self.gLength, self.iLength, self.fitnessFunc, self.baseVal, self.varVal)

		try:
			for i in range(self.gLength):
				Child.y[i] = alpha*self.y[i] + (1-alpha)*Other.y[i]
		except IndexError:
			print " !! Individual must the same length for proper Recombination !! "
		
		Child.birth()
		return Child

 	def __iadd__(self,Other):
		'''
		Overloading of the iADD operator for the incrementation of age
		'''
		self.age += Other
		return self

	def fitnessEval(self):
		if not self.fitnessComputed:
			try:			
				self.fitness = self.fitnessFunc(self.x_int, self.y_int)
				self.fitnessComputed = 1
			except:
				self.fitness = inf


class Population:
	'''
	Population Class
	'''	

	def __init__(self, nbrIndividual, genLength, indLength, func, baseVal = 0.0, varVal = 1.0, seed = 0):

		self.length = nbrIndividual
		self.rankingComputed = 0
		self.fitness = numpy.zeros(nbrIndividual, float)

		if type(seed) == list:
			self.Ind = [Individual(genLength, indLength, func, baseVal, varVal)]
			for i in range(len(seed)):
				self.Ind.append(Individual(genLength, indLength, func, baseVal, varVal))
				self.Ind[i] = seed[i]
			for i in range(nbrIndividual-len(seed)):
				self.Ind.append(Individual(genLength, indLength, func, baseVal, varVal))

		else:
			self.Ind = [Individual(genLength, indLength, func, baseVal, varVal)]

			for i in range(nbrIndividual-1):
				self.Ind.append(Individual(genLength, indLength, func, baseVal, varVal))

	def __str__(self):
		msg  = "**********"
		msg += "\nThis population have " + str(self.length) + " individuals"
		msg += "\nEach individual is " + str(self.Ind[0].iLength) + " long with a genotype length of " + str(self.Ind[0].gLength)
		msg += "\n**********"
		return msg

	def info(self):
		print "**********"
		print "This population have " + str(self.length) + " individuals"
		print "Each individual is " + str(self.Ind[0].iLength) + " long with a genotype length of " + str(self.Ind[0].gLength)
		print "**********"

	def plotAll(self):
		for i in range(self.length):
			self.Ind[i].plotIndividual()

	def rankingEval(self):
		'''
		Sorting the pop. base of the fitnessEval result
		'''
		fitnessAll = numpy.zeros(self.length)
		fitnessNorm = numpy.zeros(self.length)

		for i in range(self.length):
			self.Ind[i].fitnessEval()
			fitnessAll[i] = self.Ind[i].fitness

		maxFitness = fitnessAll.max()
		for i in range(self.length):
			fitnessNorm[i] = (maxFitness - fitnessAll[i]) / maxFitness
		fitnessSorted = fitnessNorm.argsort()

		# Compute the selection probabilities of each individual
		probability = numpy.zeros(self.length)
		S = 2.0
		for i in range(self.length):
			probability[fitnessSorted[i]] = ((2-S)/self.length) + (2*i*(S-1))/(self.length*(self.length-1))
		self.rankingComputed = 1
		self.fitness = 	fitnessAll

		return [fitnessAll, fitnessSorted[::-1], probability]

	def sortedbyAge(self):
		'''
		Sorting the pop. base of the age
		'''
		ageAll = numpy.zeros(self.length)
		for i in range(self.length):
			ageAll[i] = self.Ind[i].age
		ageSorted = ageAll.argsort()
		return ageSorted[::-1]

	def RWSelection(self, mating_pool_size):
		'''
		Make Selection of the mating pool with the roulette wheel algorithm
		'''
		A = numpy.zeros(self.length)
		mating_pool = numpy.zeros(mating_pool_size)
	
		[F,S,P] = self.rankingEval()
		
		P_Sorted = numpy.zeros(self.length)
		for i in range(self.length):
			P_Sorted[i] = P[S[i]]
	
		for i in range(self.length):
			A[i] = P_Sorted[0:(i+1)].sum()

		i = 0
		j = 0
		while j < mating_pool_size:
			r = numpy.random.random()
			i = 0
			while A[i] < r:
				i += 1
			if numpy.shape(numpy.where(mating_pool==i))[1] == 0:
				mating_pool[j] =S[i]
				j += 1

		return mating_pool

	def SUSSelection(self, mating_pool_size):
		'''
		Make Selection of the mating pool with the
		stochastic universal sampling algorithm
		'''
		A = numpy.zeros(self.length)
		mating_pool = numpy.zeros(mating_pool_size)
		r = numpy.random.random()/float(mating_pool_size)
		[F,S,P] = self.rankingEval()
		
		P_Sorted = numpy.zeros(self.length)
		for i in range(self.length):
			P_Sorted[i] = P[S[i]]
	
		for i in range(self.length):
			A[i] = P_Sorted[0:(i+1)].sum()

		i = 0
		j = 0
		while j < mating_pool_size:
			i = 0
			while A[i] <= r:
				i += 1
			mating_pool[j] = S[i]
			j += 1 
			r += (1/float(mating_pool_size))

		return mating_pool

 	def __add__(self,Other):
		'''
		Overloading of the ADD operator for merging two populations
		'''
		newLength = self.length + Other.length
		
		if (self.Ind[0].iLength == Other.Ind[0].iLength) & (self.Ind[0].gLength == Other.Ind[0].gLength):
			newPop = Population(newLength, self.Ind[0].gLength, self.Ind[0].iLength, self.Ind[0].fitnessFunc, self.Ind[0].baseVal, self.Ind[0].varVal)

			for i in range(self.length):
				newPop.Ind[i] = self.Ind[i]

			for i in range(Other.length):
				newPop.Ind[i + self.length] = Other.Ind[i]

		else:
			raise IndexError, "The two population should have the same individual length"

		return newPop

 	def __iadd__(self,Other):
		'''
		Overloading of the iADD operator for the incrementation of age
		of the whole population
		'''
		for i in range(self.length):
			self.Ind[i].age += int(Other)
		return self


class splineGA:
	'''
	Spline-based genetic optimization class
	'''	
	def __init__(self, pop):
		if isinstance(pop, Population):
			self.initPop = pop
			self.popSize = pop.length
			self.gLength = pop.Ind[0].gLength
			self.iLength = pop.Ind[0].iLength
			self.fitnessFunc = pop.Ind[0].fitnessFunc
			self.baseVal = pop.Ind[0].baseVal
			self.varVal = pop.Ind[0].varVal
		else:
			raise TypeError, 'The input should be a Population instance'
	
	def run(self, nbrGeneration, olderSize, selecSize, mutationsNbr = 1, mutationStrength = 0.1, selecMethod='SUSSelection', verbose=False):
		p = zeros(nbrGeneration)
		statMeanFitness = zeros(nbrGeneration)
		statMeanFitnessTop10 = zeros(nbrGeneration)

		Generation = range(nbrGeneration+1)
		#presentGeneration = Population(self.popSize, self.gLength, self.iLength, self.fitnessFunc, self.baseVal, self.varVal)
		presentGeneration = self.initPop
		futurGeneration = Population(self.popSize, self.gLength, self.iLength, self.fitnessFunc, self.baseVal, self.varVal)
		OffSpring = range(selecSize/2)

		archiveBestInd = zeros([nbrGeneration,  self.iLength])


		for g in range(nbrGeneration):
			'''
			Main GA optimization loop
			'''

			Selection = {
			  'SUSSelection': lambda: presentGeneration.SUSSelection(selecSize),
			  'RWSelection': lambda: presentGeneration.RWSelection(selecSize),
			}[selecMethod]()

			for i in range(selecSize/2):
				OffSpring[i] = presentGeneration.Ind[int(Selection[selecSize/2-i])] + presentGeneration.Ind[int(Selection[selecSize/2+i])]
				#mutationStrength = numpy.random.random(1)[0]/2
				OffSpring[i].mutations(mutationsNbr, mutationStrength)
				OffSpring[i].fitnessEval()
	
			futurGeneration = copy.deepcopy(presentGeneration)
			ageSorted = presentGeneration.sortedbyAge()
			[F,S,P] = presentGeneration.rankingEval()

			# Replace the oldest-sickest one by the offspring
			olderFitness = zeros([2,olderSize])
			for i in range(olderSize):
				olderFitness[0,i] = presentGeneration.Ind[ageSorted[i]].fitness
				olderFitness[1,i] = ageSorted[i]
			olderFitSorted = olderFitness[0,:].argsort()[::-1]

			tobeReplaced = zeros(selecSize/2)
			for i in range(selecSize/2):
				tobeReplaced[i] = olderFitness[1,olderFitSorted[i]]

			for j in range(selecSize/2):
				futurGeneration.Ind[int(tobeReplaced[j])] = OffSpring[j]
				futurGeneration.Ind[j].age = 0

			statMeanFitness[g] = F.mean()
	
			# Compute the mean fitness of the top10 
			for topi in arange(10) :
				statMeanFitnessTop10[g] = statMeanFitnessTop10[g] + presentGeneration.Ind[S[topi]].fitness
			statMeanFitnessTop10[g] = statMeanFitnessTop10[g] / 10.0

			archiveBestInd[g] = presentGeneration.Ind[S[0]].y_int
			# Add 1 to the age of this generation
			futurGeneration += 1
			presentGeneration = copy.deepcopy(futurGeneration)
		
			# Print some info
			if verbose:
				print('Generation ' + str(g))
				print('Fitness Mean: ' + str(statMeanFitness[g]))
				print('Fitness Variance: '+str(F.var()))
				print('* * *')
		
		return [presentGeneration, archiveBestInd, statMeanFitness, S]


class splineRelaxGA:
	'''
	Perform multiple sucessives genetic optimization with relaxed parameters
	'''	

	def __init__(self, pop):
		if isinstance(pop, Population):
			self.initPop = pop
			self.popSize = pop.length
			self.gLength = pop.Ind[0].gLength
			self.iLength = pop.Ind[0].iLength
			self.fitnessFunc = pop.Ind[0].fitnessFunc
			self.baseVal = pop.Ind[0].baseVal
			self.varVal = pop.Ind[0].varVal
		else:
			raise TypeError, 'The input should be a Population instance'

	def run(self, nbrGeneration, olderSize, selecSize, mutationsNbr = 1,
		 		maxStrength = 0.1, nbrStep = 1, selecMethod='SUSSelection', verbose=False):
		
		# Initiate empty list for storing splineGA instances
		sim = []
		sim.append(splineGA(self.initPop))
		F = zeros([nbrStep+1,nbrGeneration], float)

		# First run
		initGeneration = Population(self.popSize, self.gLength, self.iLength, self.fitnessFunc, self.baseVal, self.varVal)
		[G, A, F[0], S] = sim[0].run(nbrGeneration, olderSize, selecSize, mutationsNbr, maxStrength, selecMethod='SUSSelection', verbose=verbose)
		statMeanFitness = zeros(0,float)
		statMeanFitness = r_[statMeanFitness, F[0]]

		# Loop over second to nth run
		for i in arange(nbrStep):
			sim.append(splineGA(G))
			[G, A, F[i+1], S] = sim[i+1].run(nbrGeneration, olderSize, selecSize, mutationsNbr, maxStrength/(i+1), selecMethod='SUSSelection')
			statMeanFitness = r_[statMeanFitness, F[i+1]]

		return [G, A, statMeanFitness, S] 


