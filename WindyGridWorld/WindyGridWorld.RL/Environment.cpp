#include "Environment.h"
#include "Logger.h"
#include <assert.h>

using namespace native::frw;

Environment::Environment():
	rows_(0), cols_(0), start_(0,0), 
	target_(0,0), current_(0,0)
{
	generate_wind(cols_);
}

Environment::Environment(
	int rows, int cols,
	int startX, int startY,
	int targetX, int targetY):
	rows_(rows), cols_(cols), start_(startX, startY),
	target_(targetX, targetY), current_(start_) 
{
	generate_wind(cols_);
}

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
		cnd.x = current_.x - 1; cnd.y = current_.y + wind_strength_[current_.x];
		break;
	case 1: // UP
		cnd.x = current_.x; cnd.y = current_.y + wind_strength_[current_.x] - 1;
		break;
	case 2: // RIGHT
		cnd.x = current_.x + 1; cnd.y = current_.y + wind_strength_[current_.x];
		break;
	case 3: // DOWN
		cnd.x = current_.x; cnd.y = current_.y + wind_strength_[current_.x] + 1;
		break;
	default:
		assert(action < 4);
	}

	restrict2gridworld(cnd);

	current_ = cnd;
}

int Environment::GetValuesNumber() const {
	return rows_ * cols_;
}

int Environment::GetCurrentAsIndex() const {

	return map_grid2line(current_.x, current_.y);
}

bool Environment::IsTerminated() {

	Logger::Instance()->Add(current_.x, current_.y);
	return current_ == target_;
}

void Environment::ResetAgent() {
	current_ = start_;
}

int Environment::map_grid2line(int x, int y) const {
	return x * rows_ + y;
}

void Environment::restrict2gridworld(Cell& cnd) {

	if (cnd.x < 0)
		cnd.x = 0;
	if (cnd.x >= cols_)
		cnd.x = cols_-1;
	if (cnd.y < 0)
		cnd.y = 0;
	if (cnd.y >= rows_)
		cnd.y =rows_-1;
}

void Environment::generate_wind(int cols) {

	wind_strength_.resize(cols);

	for (int i = 0; i < cols; ++i) {
		wind_strength_[i] = 0;//i % 3 - 1;
	}
}