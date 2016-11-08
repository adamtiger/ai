#include "LearningFramework.h"

void native::LearningFramework::InitFramework(
	int type,                   // the type of the rl algorithm
	int rows, int columns,      // the size of the world
	int numofEpisodes,          // number of learning episodes
	int startX, int startY,     // coordinates of the starting cell
	int targetX, int targetY,   // coordinates of the target cell
	double alpha,               // the learning rate
	double gamma				// in case of discounted reward); 
) 
{
	
}

void native::LearningFramework::Learn() {

}

double native::LearningFramework::GetProgress() {
	return 0;
}

int native::LearningFramework::GetPathLength() {
	return 0;
}

int native::LearningFramework::GetEpisodeId() {
	return 0;
}

int native::LearningFramework::GetCoordX(int idx) {
	return 0;
}

int native::LearningFramework::GetCoordY(int idx) {
	return 0;
}

