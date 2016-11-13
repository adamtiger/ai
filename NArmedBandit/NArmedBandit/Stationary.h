#ifndef __STATIONARY__
#define __STATIONARY__

#include <vector>

namespace bandit {
	namespace stationary {

		struct NormalDistribution {

		public:

			NormalDistribution();
			NormalDistribution(float mu, float szigma);

			float Generate() const;

		private:

			float mu_;
			float szigma_;

		};


		class Stationary {

		public:

			explicit Stationary();
			~Stationary();

			void DoTrial();

			float GetValue(int idx) const;

		private:

			int choose_action() const;

			float observe_reward(int action) const;

			void refresh_values(int action, float reward);

			// machines to rob
			std::vector<NormalDistribution*> machines_;

			// action-value vector
			std::vector<float> action_value_;

			// the number of visits
			std::vector<int> visits_;

			// number of machines
			int num_;
		};

	} // stationary
} // bandit
#endif // __STATIONARY
