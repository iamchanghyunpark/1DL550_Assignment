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

#include <stdlib.h>

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

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

// finns det nagot battre satt att launcha en vektor med tradar
// hur resonerar vi kring olika modeller kring hur vi paralleliserar
// vad raknas som tva separata verisioner for fraga B

void Ped::Model::tick()
{
	int option = 1;
	switch(option) {
		case 0: { //SERIAL
				for (Tagent* a : agents) {
					a->computeNextDesiredPosition();
					a->setX( a->getDesiredX() );
					a->setY( a->getDesiredY() );
				}
				break;
				}
		case 1: { //C++ Threads
				// lambda function given to a thread
				auto f = [](std::vector<Ped::Tagent*> agents, int first, int last) -> void {
					for (int i = first; i < last; i++) {
						Tagent* a = agents[i];
						a->computeNextDesiredPosition();
						a->setX( a->getDesiredX() );
						a->setY( a->getDesiredY() );
					}
				};

				int num_threads = 2;
				int thread_work = agents.size()/num_threads;
				int curr_work_index = 0; // to keep track of "how much work"/what agents have been assigned to threads
				
				std::vector<std::thread> threads;	

				for (int i = 0; i < num_threads; i++)
				{
					// edge case when there's a request for only one thread or the last thread, in this case perform the remaining work
					if (i == num_threads-1)
					{
						std::thread t(f, agents, curr_work_index, agents.size());
						threads.push_back(std::move(t)); 
					}
					else
					{
						std::thread t(f, agents, curr_work_index, thread_work);
						threads.push_back(std::move(t));
						curr_work_index += thread_work;
					}
				} 			
	
				for (int i = 0; i < threads.size(); i++) {
					threads[i].join();
				}
				break;
			}
		case 2: { //OpenMP
				#pragma omp parallel for
					for (int i = 0; i < agents.size(); i++) {
						Tagent* a = agents[i];
						a->computeNextDesiredPosition();
						a->setX( a->getDesiredX() );
						a->setY( a->getDesiredY() );
					}
				break;		
		}
		default:
			break;	
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
