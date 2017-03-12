#ifndef __ENVIRONMENT__
#define __ENVIRONMENT__

#include <vector>

namespace bandit {

	struct NormalDistribution {

	public:

		NormalDistribution();
		NormalDistribution(float mu, float szigma);

		float Generate() const;

		void ModifyMu(float delta_mu);

		float GetMu() const;

	private:

		float mu_;
		float szigma_;
	};


	class BaseEnvironment {

	public:

		BaseEnvironment();
		~BaseEnvironment();

		float GetValue(int idx) const;

		void DoTrial();

	protected:

		int choose_action() const;

		float observe_reward(int action) const;

		virtual void refresh_values(int action, float reward) = 0;

		// machines to rob
		std::vector<NormalDistribution*> machines_;

		// action-value vector
		std::vector<float> action_value_;

		// the number of visits
		std::vector<int> visits_;

		// number of machines
		int num_;

	};

} // bandit

#endif // __ENVIRONMENT__