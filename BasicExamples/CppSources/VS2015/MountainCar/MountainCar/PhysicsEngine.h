#ifndef __PHYSICS_ENGINE__
#define __PHYSICS_ENGINE__

/*
	This is the simulator of the car on a mountain slope.
	The shape of the slope is a sine function. 
	In fact, the car is at the bottom of a valley as a starting position.
	The car can use full throttle forward or backward.
*/

class PhysicsEngine {

public:

	static PhysicsEngine* CreateDefault(double a_car);
	static PhysicsEngine* CreateWithTimeResolution(double a_car, double delta_t);

	void ExecutingAction(int action);

	double GetCurrentX();
	double GetCurrentXdot(); // horizontal velocity
	double GetCurrentY();

	double CalcMaxSpeed(); 

	void ResetSystem();

private:

	PhysicsEngine(double a_car, double delta_t);

private:

	double x_;
	double x_dot_;
	double y_;

	double a_car_; // acceleration of the car
	double delta_t_;
};

#endif // __PHYSICS_ENGINE__
