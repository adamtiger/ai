#include "Q.h"

using namespace native::alg;

Q::Q(double alpha, double gamma):
	alpha_(alpha), gamma_(gamma)
{
}

void Q::DoOneLearningIterate(){

	env_->ResetAgent();

	int inner_loop = 0;

	while (!env_->IsTerminated() && inner_loop < 100) {

		int s0 = env_->GetCurrentAsIndex();
		int a0 = 0;
		ActionSelectors::EpsilonGreedy(max(s0), &a0);

		env_->ExecuteAction(a0);
		double reward = env_->RewardFunc();

		int s1 = env_->GetCurrentAsIndex();
		int a1 = max(s1);

		qsa_[s0][a0] = (1 - alpha_) * qsa_[s0][a0] + alpha_ * (reward + gamma_ * qsa_[s1][a1]);

		inner_loop += 1;
	}
}

void Q::SetEnvironment(native::frw::Environment * env){
	env_ = env;

	qsa_.resize(env_->GetValuesNumber());

	for (int i = 0; i < env_->GetValuesNumber(); ++i) {
		qsa_[i].resize(4); // four actions are possible
	}

	for (int i = 0; i < env_->GetValuesNumber(); ++i) {
		for (int j = 0; j < 4; ++j) {
			qsa_[i][j] = 0.0;
		}
	}
}
int Q::max(int idx) const{

	int place_max = 0;

	for (int i = 1; i < 4; ++i) {
		if (qsa_[idx][i] > qsa_[idx][place_max])
			place_max = i;
	}

	return place_max;
}

