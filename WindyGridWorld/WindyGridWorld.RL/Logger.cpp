#include "Logger.h"
#include <assert.h>

using namespace native::frw;

Logger* Logger::instance_ = 0;

Logger::Logger():
	is_active_(false), epsId_(-1)
{
}

Logger* Logger::Instance() {

	if (instance_ == 0) {
		instance_ = new Logger();
	}
	
	return instance_;
}

void Logger::Activate(int epsId) {
	is_active_ = true;
	epsId_ = epsId;
}

void Logger::Passivate(){
	is_active_ = false;
	epsId_ = -1;
	path_x_coords_.clear();
	path_y_coords_.clear();
}

void Logger::Add(int x, int y) {
	if (is_active_) {
		path_x_coords_.push_back(x);
		path_y_coords_.push_back(y);
	}
}

int Logger::GetEpisodeId() const{
	assert(epsId_ != -1);
	return epsId_;
}

int Logger::GetPathLength() const{
	assert(path_x_coords_.size() == path_y_coords_.size());
	return path_x_coords_.size();
}

int Logger::GetXcoord(int idx) const {
	assert(path_x_coords_.size() > idx);
	return path_x_coords_[idx];
}

int Logger::GetYcoord(int idx) const {
	assert(path_y_coords_.size() > idx);
	return path_y_coords_[idx];
}
