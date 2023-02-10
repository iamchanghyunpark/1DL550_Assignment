//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_waypoint.h"
#include <math.h>

#include <stdlib.h>

Ped::Tagent::Tagent(int posX, int posY) {
	Ped::Tagent::init(posX, posY);
}

Ped::Tagent::Tagent(double posX, double posY) {
	Ped::Tagent::init((int)round(posX), (int)round(posY));
}

void Ped::Tagent::init(int posX, int posY) {
	x = posX;
	y = posY;
	destination = NULL;
	lastDestination = NULL;
}

int Ped::Tagent::getAllX() { return (int) allX[id]; }
int Ped::Tagent::getAllY() { return (int) allY[id]; }
float Ped::Tagent::getDestX() { return destination->getx(); }
float Ped::Tagent::getDestY() { return destination->gety(); }
float Ped::Tagent::getDestR() { return destination->getr(); }


void Ped::Tagent::destInit() { destination = waypoints.front(); }

void Ped::Tagent::updateDest(){
	waypoints.pop_front();
	waypoints.push_back(destination);
	destination = waypoints.front();
	destX[id] = destination->getx();
	destY[id] = destination->gety();
	destR[id] = destination->getr();
}

void Ped::Tagent::computeNextDesiredPosition() {
	destination = getNextDestination();
	if (destination == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}
	double diffX = destination->getx() - x;
	double diffY = destination->gety() - y;
	double len = sqrt(diffX * diffX + diffY * diffY);
	desiredPositionX = (int) round(x + diffX / len);
	desiredPositionY = (int) round(y + diffY / len);
}

void Ped::Tagent::addWaypoint(Twaypoint* wp) {
	waypoints.push_back(wp);
}

Ped::Twaypoint* Ped::Tagent::getNextDestination() {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (destination != NULL) {
		// compute if agent reached its current destination
		double diffX = destination->getx() - x;
		double diffY = destination->gety() - y;
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < destination->getr();		
	}

	if ((agentReachedDestination || destination == NULL) && !waypoints.empty()) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		waypoints.push_back(destination);
		nextDestination = waypoints.front();
		waypoints.pop_front();
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = destination;
	}

	return nextDestination;
}

// VEC FUNCTIONS

// void Ped::Tagent::computeNextDesiredPositionVec(float *xdes, float *ydes, float *x, float *y) {
// 	destination = getNextDestinationVec(xdes, ydes, x, y);
// 	if (destination == NULL) {
// 		// no destination, no need to
// 		// compute where to move to
// 		return;
// 	}

// 	__m128 X, Y, Xdes, Ydes, Xdiff, Ydiff, Xmul, Ymul, add, sqrt;	
// 	X = _mm_load_ps(&xdes[i]);
// 	Y = _mm_load_ps(&ydes[i]);
// 	Xdes = _mm_load_ps(&x[i]); // load 4 elements of x in X
// 	Ydes = _mm_load_ps(&y[i]);
// 	// double diffX = destination->getx() - X;
// 	// double diffY = destination->gety() - Y;
// 	Xdiff = _mm_sub_ps(X, Xdes);
// 	Ydiff = _mm_sub_ps(Y, Ydes);
// 	// double len = iffX * diffX + diffY * diffY);
// 	Xmul = _mm_mul_ps(Xdiff, Xdiff);
// 	Ymul = _mm_mul_ps(Ydiff, Ydiff);
// 	add = _mm_add_ps(Xmul, Ymul);
// 	sqrt = _mm_sqrt_ps(add);
// 	// desiredPositionY = (int)round(y + diffY / len);
// 	// desiredPositionX = (int)round(x + diffX / len);
// 	Xdes = _mm_add_ps(X, Xdiff);
// 	Ydes = _mm_add_ps(Y, Ydiff);
// 	X = _mm_div_ps(Xdes, sqrt);
// 	Y = _mm_div_ps(Ydes, sqrt);
// 	Xdes = _mm_round_ps(X, 0);
// 	Ydes = _mm_round_ps(Y, 0);
// }

// std::vector<Ped::Twaypoint*> Ped::Tagent::getNextDestinationVec(float *xdes, float *ydes, float *x, float *y) {
// 	Ped::Twaypoint* nextDestination = NULL;
// 	bool agentReachedDestination = false;
// 	// float *dests = (float *) _mm_malloc(agents.size() * sizeof(float), 16);

// 	if (destination != NULL) {
// 		// compute if agent reached its current destination
// 		// double diffX = destination->getx() - x;
// 		// double diffY = destination->gety() - y;
// 		// double length = sqrt(diffX * diffX + diffY * diffY);
// 		__m128 X, Y, Xdes, Ydes, Xdiff, Ydiff, Xmul, Ymul, add, sqrt;	
// 		X = _mm_load_ps(&xdes[i]);
// 		Y = _mm_load_ps(&ydes[i]);
// 		Xdes = _mm_load_ps(&x[i]); // load 4 elements of x in X
// 		Ydes = _mm_load_ps(&y[i]);
// 		// double diffX = destination->getx() - X;
// 		// double diffY = destination->gety() - Y;
// 		Xdiff = _mm_sub_ps(X, Xdes);
// 		Ydiff = _mm_sub_ps(Y, Ydes);
// 		// double len = iffX * diffX + diffY * diffY);
// 		Xmul = _mm_mul_ps(Xdiff, Xdiff);
// 		Ymul = _mm_mul_ps(Ydiff, Ydiff);
// 		add = _mm_add_ps(Xmul, Ymul);
// 		sqrt = _mm_sqrt_ps(add);
// 		for( int i = 0; i < 4; i++){
// 			agentReachedDestination = sqrt[i] < destination->getr();
// 			if (agentReachedDestination && !waypoints.empty()) {
// 				// Case 1: agent has reached destination (or has no current destination);
// 				// get next destination if available
// 				waypoints.push_back(destination);
// 				nextDestination = waypoints.front();
// 				waypoints.pop_front();
// 			}

// 			if ((agentReachedDestination || destination == NULL) && !waypoints.empty()) {
// 				// Case 1: agent has reached destination (or has no current destination);
// 				// get next destination if available
// 				waypoints.push_back(destination);
// 				nextDestination = waypoints.front();
// 				waypoints.pop_front();
// 			}

// 			else {
// 				// Case 2: agent has not yet reached destination, continue to move towards
// 				// current destination
// 				nextDestination = destination;
// 			}
	
// 		}

// 	}

// 	return nextDestination;
// }

