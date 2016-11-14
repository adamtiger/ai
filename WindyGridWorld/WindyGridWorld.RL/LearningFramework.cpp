#include "LearningFramework.h"
#include "Logger.h"
#include <math.h>

// Algorithms
#include "Q.h"
#include "Sarsa.h"

using namespace native::frw;
using namespace native::alg;
using namespace native;

LearningFramework::~LearningFramework() {
	delete alg;
}

void LearningFramework::InitFramework(
	int type,                   // the type of the rl algorithm
	int rows, int columns,      // the size of the world
	int numofEpisodes,          // number of learning episodes
	int startX, int startY,     // coordinates of the starting cell
	int targetX, int targetY,   // coordinates of the target cell
	double alpha,               // the learning rate
	double gamma				// in case of discounted reward); 
) 
{
	iter = 0;
	numEps = numofEpisodes;

	Environment* env = new Environment(rows, columns, startX, startY, targetX, targetY);

	switch (type) {
	case 0:
		alg = new Q(alpha, gamma);
		break;
	case 1:
		alg = new Sarsa(alpha, gamma);
		break;
	}

	alg->SetEnvironment(env);
}

void LearningFramework::Learn() {

	Logger::Instance()->Passivate();

	int burst = (int)std::round(numEps * 0.1);

	for (int i = 0; i < burst - 1; ++i) {
		alg->DoOneLearningIterate();
	}

	Logger::Instance()->Activate(iter + burst - 1);
	alg->DoOneLearningIterate();

	iter += burst;
}

double LearningFramework::GetProgress() {
	return iter/numEps;
}

int LearningFramework::GetPathLength() {
	return Logger::Instance()->GetPathLength();
}

int LearningFramework::GetEpisodeId() {
	return Logger::Instance()->GetEpisodeId();
}

int LearningFramework::GetCoordX(int idx) {
	return Logger::Instance()->GetXcoord(idx);
}

int LearningFramework::GetCoordY(int idx) {
	return Logger::Instance()->GetYcoord(idx);
}

