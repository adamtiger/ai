#include "PhysicsEngine.h"

#include <math.h>

using namespace std;

#define DELTA_T 0.01
#define G       9.81
#define PI      3.14

PhysicsEngine* PhysicsEngine::CreateDefault(double a_car){

	return new PhysicsEngine(a_car, DELTA_T);
}

PhysicsEngine * PhysicsEngine::CreateWithTimeResolution(double a_car, double delta_t){

	return new PhysicsEngine(a_car, delta_t);
}

void PhysicsEngine::ExecutingAction(int action){ // action == 0 : accelerate to left, 1 : to right

	double a = action == 1 ? a_car_ : -a_car_;
	double tg_alpha = cos(x_ - PI / 2);
	double cos_alpha = 1.0 / sqrt(1 + tg_alpha * tg_alpha);
	double sin_alpha = tg_alpha * cos_alpha;

	double a_tx = (a - G * sin_alpha) * cos_alpha;

	x_ = x_ + x_dot_ * delta_t_;
	x_dot_ = x_dot_ + a_tx * delta_t_;
}

double PhysicsEngine::GetCurrentX(){

	return x_;
}

double PhysicsEngine::GetCurrentXdot(){

	return x_dot_;
}

double PhysicsEngine::getCurrentY(){

	return sin(x_ - PI / 2.0) + 1;
}

double PhysicsEngine::CalcMaxSpeed(){

	return sqrt(2 * G + a_car_ * sqrt(2) * PI);
}

void PhysicsEngine::ResetSystem(){
	x_ = 0.0;
	x_dot_ = 0.0;
	y_ = 0.0;
}

PhysicsEngine::PhysicsEngine(double a_car, double delta_t) {

	a_car_ = fabs(a_car);
	delta_t_ = delta_t;

	x_ = 0.0;
	x_dot_ = 0.0;
}
