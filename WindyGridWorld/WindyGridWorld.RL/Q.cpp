#include "Q.h"

using namespace native::alg;

Q::Q(double alpha, double gamma):
	frw::IAlgorithm(alpha, gamma)
{
}

void Q::DoOneLearningIterate(){

	env_->ResetAgent();
	int s0 = env_->GetCurrentAsIndex();

	int inner_loop = 0;

	while (!env_->IsTerminated() && inner_loop < 100) {

		int a0 = 0;
		ActionSelectors::EpsilonGreedy(max(s0), &a0);

		env_->ExecuteAction(a0);
		double reward = env_->RewardFunc();

		int s1 = env_->GetCurrentAsIndex();
		int a1 = max(s1);

		qsa_[s0][a0] = (1 - alpha_) * qsa_[s0][a0] + alpha_ * (reward + gamma_ * qsa_[s1][a1]);

		s0 = s1;
		inner_loop += 1;
	}
}


