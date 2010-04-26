genSpline - Spline-based genetic optimization class
===================================================
The genSpline python package is a Spline-based genetic optimization class.


Requirements
---------------------------------------------------
scipy
numpy
matplotlib


Installation
---------------------------------------------------
git clone git://github.com/mlaprise/genSpline.git
cd genSpline
sudo python setup.py install


Example
---------------------------------------------------

Simple Hello world example ::

	from genSpline import *
	import matplotlib.pyplot as plt
	 
	# Parametre du GA
	older = 10
	popSize = 30
	select = 5
	gen = 32
	longueur = 256
	nbrGenerations = 200
	mutations = 10
	nbrStep = 50
	 
	def fitness(x_int, y_int):
		return exp(y_int.var())
	 
	presentGeneration = Population(popSize, gen, longueur, fitness, 0.0, 1.0,
	splineType='real')
	 
	sim1 = splineRelaxGA(presentGeneration)
	 
	[presentGeneration, archiveBestInd, statMeanFitness, S] = sim1.run(nbrGenerations,
	older, select, mutations, 0.12, nbrStep, selecMethod='SUSSelection')
	 
	presentGeneration.Ind[S[0]].plot()
