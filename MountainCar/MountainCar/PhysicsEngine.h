#ifndef __PHYSICS_ENGINE__
#define __PHYSICS_ENGINE__



class PhysicsEngine {

public:

	static PhysicsEngine* CreateDefault(double a_car);
	static PhysicsEngine* CreateWithTimeResolution(double a_car, double delta_t);

	void ExecutingAction(int action);

	double GetCurrentX();
	double GetCurrentXdot();
	double getCurrentY();

	double CalcMaxSpeed();

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
