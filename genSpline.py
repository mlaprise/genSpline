#!/usr/bin/env python


'''
	
	Spline-based genetic optimization class



	Author: 	Martin Laprise
		    	Universite Laval
				martin.laprise.1@ulaval.ca
                 
'''


import numpy
import numpy.random
from numpy import *
import scipy
import pylab as pl
import scipy.interpolate
from scipy import integrate
from scipy import signal
import PyOFTK
from PyOFTK.utilities import *
import time
import copy


class Individual:

	def __init__(self, geneLength, individualLength, func, baseVal = 0.0, varVal = 10.0):
		
		# Constructor
		self.length = 0
		self.fitness = 0.0
		self.fitnessFunc = func
		self.fitnessComputed = 0
		self.age = 0
		self.xBound = [-1,1]
		self.gLength = geneLength
		self.iLength = individualLength

		self.x = pl.linspace(-16,16, geneLength)
		self.y = numpy.zeros(geneLength, complex)
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
		Child = Individual(self.gLength, self.iLength, self.fitnessFunc)

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

	def __init__(self, nbrIndividual, genLength, indLength, func, seed = 0):

		self.length = nbrIndividual
		self.rankingComputed = 0

		if type(seed) == list:
			self.Ind = [Individual(genLength, indLength, func, 0)]
			for i in range(len(seed)):
				self.Ind.append(Individual(genLength, indLength, func, 0))
				self.Ind[i] = seed[i]
			for i in range(nbrIndividual-len(seed)):
				self.Ind.append(Individual(genLength, indLength, func, 0))

		else:
			self.Ind = [Individual(genLength, indLength, func)]

			for i in range(nbrIndividual-1):
				self.Ind.append(Individual(genLength, indLength, func))

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
			newPop = Population(newLength, self.Ind[0].gLength, self.Ind[0].iLength)

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

	def __init__(self, pop):
		if isinstance(pop, Population):
			self.popSize = pop.length
			self.gLength = pop.Ind[0].gLength
			self.iLength = pop.Ind[0].iLength
			self.fitnessFunc = pop.Ind[0].fitnessFunc
		else:
			raise TypeError, 'The input should be a Population instance'
	
	def run(self, nbrGeneration, olderSize, selecSize, mutationsNbr = 1, mutationStrength = 0.1):
		p = zeros(nbrGeneration)
		statMeanFitness = zeros(nbrGeneration)
		statMeanFitnessTop10 = zeros(nbrGeneration)

		Generation = range(nbrGeneration+1)
		presentGeneration = Population(self.popSize, self.gLength, self.iLength, self.fitnessFunc)
		futurGeneration = Population(self.popSize, self.gLength, self.iLength, self.fitnessFunc)
		OffSpring = range(selecSize/2)

		archiveBestInd = zeros([nbrGeneration,  self.iLength])


		for g in range(nbrGeneration):
			'''
			Main GA optimization loop
			'''

			Selection = presentGeneration.SUSSelection(selecSize)

			for i in range(selecSize/2):
				OffSpring[i] = presentGeneration.Ind[int(Selection[selecSize/2-i])] + presentGeneration.Ind[int(Selection[selecSize/2+i])]
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
		
		return [presentGeneration, archiveBestInd, statMeanFitness, S]






