#include "Environment.h"
#include "Logger.h"
#include <assert.h>

using namespace native::frw;

Environment::Environment():
	rows_(0), cols_(0), start_(0,0), 
	target_(0,0), current_(0,0){}

Environment::Environment(
	int rows, int cols,
	int startX, int startY,
	int targetX, int targetY):
	rows_(rows), cols_(cols), start_(startX, startY),
	target_(targetX, targetY), current_(start_) {}

double Environment::RewardFunc() const{

	double rw;

	if (current_ == target_) {
		rw = 100;
	}
	else {
		rw = -1;
	}

	return rw;
}

void Environment::ExecuteAction(int action) {

	Cell cnd;

	switch (action) {
	case 0: // LEFT
		cnd.x = current_.x - 1; cnd.y = current_.y + wind_strength_[cnd.y];
		break;
	case 1: // UP
		cnd.x = current_.x; cnd.y = current_.y + wind_strength_[cnd.y] - 1;
		break;
	case 2: // RIGHT
		cnd.x = current_.x + 1; cnd.y = current_.y + wind_strength_[cnd.y];
		break;
	case 3: // DOWN
		cnd.x = current_.x; cnd.y = current_.y + wind_strength_[cnd.y] + 1;
		break;
	default:
		assert(action < 4);
	}

	restrict2gridworld(cnd);

	current_ = cnd;

	Logger::Instance()->Add(cnd.x, cnd.y);
}

int Environment::GetX() const{
	return current_.x;
}

int Environment::GetY() const{
	return current_.y;
}

void Environment::restrict2gridworld(Cell& cnd) {

	if (cnd.x < 0)
		cnd.x = 0;
	if (cnd.x >= cols_)
		cnd.x = cols_;
	if (cnd.y < 0)
		cnd.y = 0;
	if (cnd.y >= rows_)
		cnd.y =rows_;
}