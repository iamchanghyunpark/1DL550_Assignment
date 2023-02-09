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

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	cout << "setup";
	// Allocate memory for x coordinates
	x = (float *) _mm_malloc(agents.size() * sizeof(float), 16);

	// Allocate memory for y coordinates
	y = (float *) _mm_malloc(agents.size() * sizeof(float), 16);

	// Allocate memory for x coordinates
	xdes = (float *) _mm_malloc(agents.size() * sizeof(float), 16);

	// Allocate memory for y coordinates
	ydes = (float *) _mm_malloc(agents.size() * sizeof(float), 16);

	// for (int i = 0; i < allAgents.size(); i++){
	// 	allAgents[i]->computeNextDesiredPosition();
	// 	x[i] = allAgents[i]->getDesiredX();
	// 	y[i] = allAgents[i]->getDesiredY();
	// }

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
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
		if(allAgents.size() % numThreads == 0)
		{
			for(int i = 0; i < numThreads; i++) {
				threads[i]= std::thread(action, allAgents, offset, (offset + agentsPerThread));
				offset = offset + agentsPerThread;
			}
		}
		else 
		{
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
	//SIMD IMPLEMENTATION
	if(this->implementation == VECTOR) {
		x = getVecX();
		y = getVecY();
		xdes = getVecXdes();
		ydes = getVecYdes();
		for (int i = 0; i < allDest.size(); i++){
			//allAgents[i]->computeNextDesiredPosition();
			xdes[i] = allDest[i]->getx();
			ydes[i] = allDest[i]->gety();	
		}
		
		for (int i = 0; i < allAgents.size(); i++){
			x[i] = allAgents[i]->getX();
			y[i] = allAgents[i]->getY();
		}

		for (int i = 0; i < allAgents.size(); i+=4){
			Ped::Twaypoint* nextDestination = NULL;
			bool agentReachedDestination = false;
			__m128 X, Y, Xdes, Ydes, Xdiff, Ydiff, Xmul, Ymul, add, sqrt;
			if (allAgents[i]->getDestination() != NULL) {
				X = _mm_load_ps(&xdes[i]);
				Y = _mm_load_ps(&ydes[i]);
				Xdes = _mm_load_ps(&x[i]); // load 4 elements of x in X
				Ydes = _mm_load_ps(&y[i]);
				// double diffX = destination->getx() - X;
				// double diffY = destination->gety() - Y;
				Xdiff = _mm_sub_ps(X, Xdes);
				Ydiff = _mm_sub_ps(Y, Ydes);
				// double len = iffX * diffX + diffY * diffY);
				Xmul = _mm_mul_ps(Xdiff, Xdiff);
				Ymul = _mm_mul_ps(Ydiff, Ydiff);
				add = _mm_add_ps(Xmul, Ymul);
				sqrt = _mm_sqrt_ps(add);
				// desiredPositionY = (int)round(y + diffY / len);
				// desiredPositionX = (int)round(x + diffX / len);
				Xdes = _mm_add_ps(X, Xdiff);
				Ydes = _mm_add_ps(Y, Ydiff);
				X = _mm_div_ps(Xdes, sqrt);
				Y = _mm_div_ps(Ydes, sqrt);
				Xdes = _mm_round_ps(X, 0);
				Ydes = _mm_round_ps(Y, 0);

				for (int j = 0; j < 4 ; j++){
					agentReachedDestination = sqrt[i] < allAgents[j]->getDestination()->getr();
					if (agentReachedDestination && !allAgents[j]->getWaypoints().empty()) {
						// Case 1: agent has reached destination (or has no current destination);
						// get next destination if available
						allAgents[j]->getWaypoints().push_back(allAgents[j]->getDestination());
						nextDestination = allAgents[j]->getWaypoints().front();
						allAgents[j]->getWaypoints().pop_front();
					}
					if ((agentReachedDestination || allAgents[j]->getDestination() == NULL) && !allAgents[j]->getWaypoints().empty()) {
						// Case 1: agent has reached destination (or has no current destination);
						// get next destination if available
						allAgents[j]->getWaypoints().push_back(allAgents[j]->getDestination());
						nextDestination = allAgents[j]->getWaypoints().front();
						allAgents[j]->getWaypoints().pop_front();
					}

					else {
						// Case 2: agent has not yet reached destination, continue to move towards
						// current destination
						nextDestination = allAgents[j]->getDestination();
					}
				}
			}
			if (allAgents[i]->getDestination() != NULL) {	
				X = _mm_load_ps(&xdes[i]);
				Y = _mm_load_ps(&ydes[i]);
				Xdes = _mm_load_ps(&x[i]); // load 4 elements of x in X
				Ydes = _mm_load_ps(&y[i]);
				// double diffX = destination->getx() - X;
				// double diffY = destination->gety() - Y;
				Xdiff = _mm_sub_ps(X, Xdes);
				Ydiff = _mm_sub_ps(Y, Ydes);
				// double len = iffX * diffX + diffY * diffY);
				Xmul = _mm_mul_ps(Xdiff, Xdiff);
				Ymul = _mm_mul_ps(Ydiff, Ydiff);
				add = _mm_add_ps(Xmul, Ymul);
				sqrt = _mm_sqrt_ps(add);
				// desiredPositionY = (int)round(y + diffY / len);
				// desiredPositionX = (int)round(x + diffX / len);
				Xdes = _mm_add_ps(X, Xdiff);
				Ydes = _mm_add_ps(Y, Ydiff);
				X = _mm_div_ps(Xdes, sqrt);
				Y = _mm_div_ps(Ydes, sqrt);
				Xdes = _mm_round_ps(X, 0);
				Ydes = _mm_round_ps(Y, 0);
			}
			for (int j = 0; j < 4 ; j++){
				allAgents[j]->setX(X[j]);
				allAgents[j]->setY(Y[j]);
				cout << "XDES" << X[j];
				cout << "\n";
				cout << "YDES" << Y[j];
				cout << "\n";
			}
		
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

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

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
}
