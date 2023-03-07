//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>
#include <math.h>

using namespace std;

namespace Ped {
	class Twaypoint;

	class Tagent {
	public:
		Tagent(int posX, int posY);
		Tagent(double posX, double posY);

		/* SETTERS */
		void setallX(float *newallX) { allX = newallX; }
		void setallY(float *newallY) { allY = newallY; }

		void setdestX(float *newdX) { destX = newdX; };
		void setdestY(float *newdY) { destY = newdY; };
		void setdestR(float *newdR) { destR = newdR; };

		void setId(int newid) { id = newid; };

		// Sets the agent's position
		void setX(int newX) { x = newX; };
		void setY(int newY) { y = newY; };

		void setX(float newX) { x = (int) round(newX); };
		void setY(float newY) { y = (int) round(newY); };

		/* GETTERS */
		// Position of agent defined by x and y
		float getX() const { return x; };
		float getY() const { return y; };

		float getDestX();
		float getDestY();
		float getDestR();

		int* getXAddr() { return &x; };
		int* getYAddr() { return &y; };

		// Returns the coordinates of the desired position
		int getDesiredX() const { return desiredPositionX; };
		int  getDesiredY() const { return desiredPositionY; };

        int waypointSize() {return waypoints.size();};

		/* FUNCTIONS */
		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint* wp);

		// Update the position according to get closer
		// to the current destination
		void computeNextDesiredPosition();

		void destInit();

		void updateDest();

	private:
		Tagent() {};

		// The agent's current position
		int x;
		int y;
		int id;

		float *allX;
		float *allY;
		float *destX;
		float *destY;
		float *destR;

		// The agent's desired next position
		int desiredPositionX;
		int desiredPositionY;

		// The current destination (may require several steps to reach)
		Twaypoint* destination;

		// The last destination
		Twaypoint* lastDestination;

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint*> waypoints;

		// Internal init function 
		void init(int posX, int posY);

		// Returns the next destination to visit
		Twaypoint* getNextDestination();
	};
}

#endif
