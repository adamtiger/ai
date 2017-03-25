#include "stdafx.h"

#include "ContainerProject.h"

// NArmedBandit functions:
#include "Stationary.h"
#include "NonStationary.h"

// MountainCar functions:
#include "LearningCoordinator.h"
#include "Logger.h"

// Qlearning functions:
#include "WindyGridWorld.h"
#include "rl_q.h"

float* StationaryAnalysis()
{
	bandit::stationary::Stationary stn;

	for (int i = 0; i < 10000; ++i)
		stn.DoTrial();

	float* values = new float[10];
	for (int i = 0; i < 10; ++i) {

		values[i] = stn.GetValue(i);
	}

	return values;
}

float* NonStationaryAnalysis()
{
	bandit::nonstationary::NonStationary nstn;

	for (int i = 0; i < 10000; ++i) {

		nstn.DoTrial();

		if (i % 500 == 0)
			nstn.ChangeEnv();
	}


	float* values = new float[10];
	for (int i = 0; i < 10; ++i) {

		values[i] = nstn.GetValue(i);
	}

	return values;
}

void SolveMountainCar()
{
	LearningCoordinator lc(10, 16, 0.2, 0.02, 0.98);

	lc.DoLearning(10000);

	lc.TestAgent();

	Logger* log = lc.GetLogger();

	log->Save2File("result.txt");
}

int* SolveWindyGridWorld()
{
	/* Initialization. */

	float alpha = 0.1f;
	float gamma = 0.95f; // Stable convergence occurs above this value.
	int width = 6; int height = 5;
	int strength[6] = { 0, 0, 1, 2, 2, 0 };

	WindyGridWorld wdw(6, 5, 1, 2, 5, 1);
	rl::q::Q q(alpha, gamma, width * height);

	wdw.SetStrength(strength, width);

	int num_episodes = 1200;
	int max_steps_per_ep = 1000;
	float decay_ratio = 1.03f;

	std::vector<int> length_of_episodes(num_episodes);

	for (int i = 0; i < num_episodes; ++i) {

		wdw.StartNewEpisode();
		while (!(wdw.IsTerminated() || wdw.IsFailed2Converge(max_steps_per_ep))) {

			const Position& pos = wdw.GetCurrent();
			rl::q::State state(pos.x * height + pos.y, pos.x, pos.y, wdw.IsTerminated());
			rl::q::Action ac = q.ObserveAction(state);
			rl::q::State nextSt = [](rl::q::tAction act, WindyGridWorld& wdw, int hg) {
				int x = 0, y = 0;
				act.get_deviation(&x, &y);
				const Position& pos = wdw.Move(x, y);

				rl::q::State state(
					pos.x * hg + pos.y,
					pos.x,
					pos.y,
					wdw.IsTerminated()
				);

				return state;
			}(ac, wdw, height); // This moves the agent as well.

			float reward = wdw.IsTerminated() ? 100 : -1;
			if (i <= 1000)
				q.UpdateActionStateValue(reward, ac, state, nextSt);
		}

		length_of_episodes[i] = wdw.LengthOfEpisode();
		if (i % 5 == 0)
			q.DecayAlpha(decay_ratio);
	}

	int* result = new int[length_of_episodes.size()];
	for (int i = 0; i < 12; ++i) {
		result[i] = length_of_episodes[i*100];
	}
		
	return result;
}


