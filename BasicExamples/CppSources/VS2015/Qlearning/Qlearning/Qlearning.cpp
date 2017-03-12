// Qlearning.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>

#include "WindyGridWorld.h"
#include "rl_q.h"

rl::q::tState NextState(rl::q::tAction act, WindyGridWorld& wdw, int hg) {

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
}


int main()
{
	/* Initialization. */

	float alpha = 0.1f;
	float gamma = 0.95f; // Stable convergence occurs above this value.
	int width = 6; int height = 5;
	int strength[6] = {0, 0, 1, 2, 2, 0};

	WindyGridWorld wdw(6, 5, 1, 2, 5, 1);
	rl::q::Q q(alpha, gamma, width * height);

	wdw.SetStrength(strength, width);

	int num_episodes = 1200;
	int max_steps_per_ep = 1000;
	float decay_ratio = 1.03f;

	/* Execute the episodes. */

	std::cout << "Start learning." << std::endl;

	std::vector<int> length_of_episodes(num_episodes);

	for (int i = 0; i < num_episodes; ++i) {

		std::cout << "Episode: " << i << "/" << num_episodes << std::endl;

		wdw.StartNewEpisode();
		while (!(wdw.IsTerminated() || wdw.IsFailed2Converge(max_steps_per_ep))) {

			const Position& pos = wdw.GetCurrent();
			rl::q::State state(pos.x * height + pos.y, pos.x, pos.y, wdw.IsTerminated());
			rl::q::Action ac = q.ObserveAction(state);
			rl::q::State nextSt = NextState(ac, wdw, height); // This moves the agent as well.

			float reward = wdw.IsTerminated() ? 100 : -1;
			if(i <= 1000)
				q.UpdateActionStateValue(reward, ac, state, nextSt);
		}

		length_of_episodes[i] = wdw.LengthOfEpisode();
		if(i % 5 == 0)
			q.DecayAlpha(decay_ratio);
	}

	std::cout << "Analise the results." << std::endl;

	for (int i = 0; i < num_episodes; ++i)
		std::cout << "Length of episode " << i << ": " << length_of_episodes[i] << std::endl;

    return 0;
}

