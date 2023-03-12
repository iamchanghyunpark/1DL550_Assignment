//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include <mm_malloc.h>
#include <stdlib.h>
#define SSE
#include <emmintrin.h>
#include <smmintrin.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Size variable
	int size = (1 + (agents.size() / 4))*4;

	// Allocate memory for coordinates
	X = (float *) _mm_malloc(size * sizeof(float), 16);
	Y = (float *) _mm_malloc(size * sizeof(float), 16);
	destX = (float *) _mm_malloc(size * sizeof(float), 16);
	destY = (float *) _mm_malloc(size * sizeof(float), 16);
	destR = (float *) _mm_malloc(size * sizeof(float), 16);
	desiredX = (int*) malloc(agents.size() * sizeof(int));
	desiredY = (int*) malloc(agents.size() * sizeof(int));
	// Initialize values of coordinates
	for (int i = 0; i < agents.size(); i++){
		agents.at(i)->setId(i);
		agents.at(i)->setallX(X);
		agents.at(i)->setallY(Y);
		agents.at(i)->setdestX(destX);
		agents.at(i)->setdestY(destY);
		agents.at(i)->setdestR(destR);
		X[i] = agents.at(i)->getX();
		Y[i] = agents.at(i)->getY();
		agents.at(i)->destInit();
		destX[i] = agents.at(i)->getDestX();
		destY[i] = agents.at(i)->getDestY();
		destR[i] = agents.at(i)->getDestR();
	}

	// Initialize regions' x values
	region1 = 40;
	region2 = 80;
	region3 = 120;
	region4 = 160;

	// Desired positions
	for (int i = 0; i < agents.size(); i++)
	{
		desiredX[i] = agents[i]->getDesiredX();
		//std::cout << type_name<decltype(desiredX[i])>() << '\n';
		desiredY[i] = agents[i]->getDesiredY();
	}

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
	setupHeatmapCuda();
}

void action(std::vector<Ped::Tagent *> allAgents, int start, int end)
{
	for (int i = start; i < end; i++) {
		allAgents[i]->computeNextDesiredPosition();
		allAgents[i]->setX(allAgents[i]->getDesiredX());
		allAgents[i]->setY(allAgents[i]->getDesiredY());
	}
}

void Ped::Model::tick()
{
	//To choose which implementation to run, change line 70 in the main file located in the demo folder
	//Change the last argument for the setup function to whichever implementation you want to run
	//Example: model.setup(parser.getAgents(), parser.getWaypoints(), Ped::SEQ);   will run sequential version
	// model.setup(parser.getAgents(), parser.getWaypoints(), Ped::OMP); will run the OpenMP version
	
	// Retrieve vector of all agents
	const std::vector<Tagent *> allAgents = getAgents();
	const std::vector<Twaypoint *> allDest = getDest();

	//C++ THREAD IMPLEMENTATION
	if(this->implementation == PTHREAD) {
		int numThreads = 4;
		int offset = 0;
		std::thread threads[numThreads];
		int agentsPerThread = (int) allAgents.size() / numThreads;
		if(allAgents.size() % numThreads == 0) {
			for(int i = 0; i < numThreads; i++) {
				threads[i]= std::thread(action, allAgents, offset, (offset + agentsPerThread));
				offset = offset + agentsPerThread;
			}
		}
		else {
			for(int i = 0; i < numThreads-1; i++) {
				threads[i]= std::thread(action, allAgents, offset, (offset + agentsPerThread));
				offset = offset + agentsPerThread;
			}

			if(offset < allAgents.size()) {
				threads[numThreads-1] = std::thread(action, allAgents, offset, allAgents.size());
			}
		}

		for(int i = 0; i < numThreads; i++) {
			threads[i].join();
		}
	}
	//SEQUENTIAL IMPLEMENTATION
	if(this->implementation == SEQ) {
		for (Tagent *agent: allAgents) {
			agent->computeNextDesiredPosition();
			agent->setX(agent->getDesiredX());
			agent->setY(agent->getDesiredY());
		}
	}
	//OPENMP IMPLEMENTATION
	if(this->implementation == OMP) {
		#pragma omp parallel for default(none) shared(allAgents) num_threads(4)
		for (Tagent *agent: allAgents) {
			agent->computeNextDesiredPosition();
			agent->setX(agent->getDesiredX());
			agent->setY(agent->getDesiredY());
		}
	}
	//VECTOR+OMP IMPLEMENTATION
	if(this->implementation == VECTOR) {
		int size = this->agents.size();
		#pragma omp parallel for default(none) shared(size) num_threads(4)
		for (int i = 0; i < size; i+=4) {
			// Vectorized implementation of the
			// computeNextPosition()-function		
			__m128 xReg, yReg, destXReg, destYReg;
			// Load elements
			xReg = _mm_load_ps(&X[i]);
			yReg = _mm_load_ps(&Y[i]);
			destXReg = _mm_load_ps(&destX[i]); 
			destYReg = _mm_load_ps(&destY[i]);
			// Get difference
			__m128 Xdiff = _mm_sub_ps(destXReg, xReg);
			__m128 Ydiff = _mm_sub_ps(destYReg, yReg);
			// Get length
			__m128 Xmul = _mm_mul_ps(Xdiff, Xdiff);
			__m128 Ymul = _mm_mul_ps(Ydiff, Ydiff);
			__m128 len = _mm_sqrt_ps(_mm_add_ps(Xmul, Ymul));
			// Calculate if agents has reached destination
			__m128 rReg = _mm_load_ps(&destR[i]);
			__m128 agentReach = _mm_cmplt_ps(len, rReg);
			int mask = _mm_movemask_ps(agentReach);
			// Update agents destination depending on ifNow we can send off race and car to be 
			// it has reached its previous destination
			for (int j = 0; j < 4; j++) {
				if (mask & 1) {
					if (i+j < size) agents.at(i+j)->updateDest();
				}
				mask >>= 1;
			}
			// Vectorized implementation of the
			// computeNextDesiredPosition()-function
			destXReg = _mm_load_ps(&destX[i]); 
			destYReg = _mm_load_ps(&destY[i]);
			// Get difference
			Xdiff = _mm_sub_ps(destXReg, xReg);
			Ydiff = _mm_sub_ps(destYReg, yReg);
			// Get length
			Xmul = _mm_mul_ps(Xdiff, Xdiff);
			Ymul = _mm_mul_ps(Ydiff, Ydiff);
			len = _mm_sqrt_ps(_mm_add_ps(Xmul, Ymul));
			// Calculate new desired postition
			__m128 Xdiv = _mm_div_ps(Xdiff, len);
			__m128 Ydiv = _mm_div_ps(Ydiff, len);
			__m128 Xadd = _mm_add_ps(xReg, Xdiv);
			__m128 Yadd = _mm_add_ps(yReg, Ydiv);
			// Round to nearest lowerbound integer
			__m128 p5 = _mm_set1_ps(0.5);
			Xadd = _mm_add_ps(Xadd, p5);
			Yadd = _mm_add_ps(Yadd, p5);
			destXReg = _mm_floor_ps(Xadd);
			destYReg = _mm_floor_ps(Yadd);
			// Store result
			_mm_store_ps(&X[i], destXReg);
			_mm_store_ps(&Y[i], destYReg);
		}
		// Iterate through all agents and
		// set their new positions
		#pragma omp parallel for default(none) shared(size) num_threads(4)
		for(int j = 0;j < size; j++) {
			agents.at(j)->setX(X[j]);
			agents.at(j)->setY(Y[j]);
		}
	}
	if(this->implementation == TASK) {
		// Desired positions of agents
		for (int i = 0; i < agents.size(); i++)
		{
			desiredX[i] = agents[i]->getDesiredX();
			desiredY[i] = agents[i]->getDesiredY();
		}
		// Move agents
		#pragma omp parallel shared(allAgents) num_threads(4) //for
		#pragma omp single 
		{
		for (Tagent *agent: allAgents) {
			int Xpos = agent-> getX();
			if(Xpos < region1) {
				#pragma omp task
				agent->computeNextDesiredPosition();
				move(agent);

			} else if(Xpos < region2) {
				#pragma omp task
				agent->computeNextDesiredPosition();
				move(agent);
		
			} else if(Xpos < region3){
				#pragma omp task
				agent->computeNextDesiredPosition();
				move(agent);
			} else {
				#pragma omp task
				agent->computeNextDesiredPosition();
				move(agent);
			}
		}
		updateHeatmapCuda();
		}

	}

	if(this->implementation == MOVESEQ) {
		for (Tagent *agent: allAgents) {
			int Xpos = agent-> getX();
			agent->computeNextDesiredPosition();
			move(agent);
		}
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.


void Ped::Model::move(Ped::Tagent *agent) 
{
	
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself	
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			if(agent->getX() < region1 && (*it).first > region1-1 || agent->getX() > region1-1 && agent->getX() < region2 && (*it).first < region1) {
				#pragma omp critical 
				{
				agent->setX((*it).first);
				agent->setY((*it).second);

				}
			} else if(agent->getX() < region2 && agent->getX() > region1 && (*it).first > region2-1 || agent->getX() > region2-1 && agent->getX() < region3 && (*it).first < region2) {
				#pragma omp critical 
				{
				agent->setX((*it).first);
				agent->setY((*it).second);

				}
			} else if(agent->getX() < region3 && agent->getX() > region2 && (*it).first > region3-1 || agent->getX() > region3-1 && (*it).first < region3) {
				#pragma omp critical 
				{
				agent->setX((*it).first);
				agent->setY((*it).second);
				}
			} else {	
				agent->setX((*it).first);
				agent->setY((*it).second);
			}

			break;
		}
	}
}

void Ped::Model::movecrit(Ped::Tagent *agent) 
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself	
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			if(agent->getX() < region1 && (*it).first > region1-1 || agent->getX() > region1-1 && agent->getX() < region2 && (*it).first < region1) {
				#pragma omp critical 
				{
				agent->setX((*it).first);
				agent->setY((*it).second);

				}
			} else if(agent->getX() < region2 && agent->getX() > region1 && (*it).first > region2-1 || agent->getX() > region2-1 && agent->getX() < region3 && (*it).first < region2) {
				#pragma omp critical 
				{
				agent->setX((*it).first);
				agent->setY((*it).second);

				}
			} else if(agent->getX() < region3 && agent->getX() > region2 && (*it).first > region3-1 || agent->getX() > region3-1 && (*it).first < region3) {
				#pragma omp critical 
				{
				agent->setX((*it).first);
				agent->setY((*it).second);
				}
			} else {	
				agent->setX((*it).first);
				agent->setY((*it).second);
			}

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
	free(*heatmap);
	free(heatmap);
	free(*scaled_heatmap);
	free(scaled_heatmap);
	free(*blurred_heatmap);
	free(blurred_heatmap);
}