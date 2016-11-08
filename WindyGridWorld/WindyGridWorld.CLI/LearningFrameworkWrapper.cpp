#include "LearningFrameworkWrapper.h"
#include "..\WindyGridWorld.RL\LearningFramework.h"

native::wrapper::LearningFrameworkWrapper::LearningFrameworkWrapper():
	lf(new LearningFramework())
{
}

void native::wrapper::LearningFrameworkWrapper::InitFramework(
	int type, int rows, int columns, int numofEpisodes, int startX, int startY,     
	int targetX, int targetY, double alpha, double gamma) 
{
	lf->InitFramework(
		type, rows, columns, numofEpisodes, startX, startY,
		targetX, targetY, alpha, gamma);
}

void native::wrapper::LearningFrameworkWrapper::Learn() {
	lf->Learn();
}

double native::wrapper::LearningFrameworkWrapper::GetProgress() {
	return lf->GetProgress();
}

int native::wrapper::LearningFrameworkWrapper::GetPathLength() {
	return lf->GetPathLength();
}

int native::wrapper::LearningFrameworkWrapper::GetEpisodeId() {
	return lf->GetEpisodeId();
}

int native::wrapper::LearningFrameworkWrapper::GetCoordX(int idx) {
	return lf->GetCoordX(idx);
}

int native::wrapper::LearningFrameworkWrapper::GetCoordY(int idx) {
	return lf->GetCoordY(idx);
}
