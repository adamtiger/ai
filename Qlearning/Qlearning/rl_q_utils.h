#ifndef __RL_Q_UTILS__
#define __RL_Q_UTILS__

#include <string>
#include <vector>

namespace rl {
	namespace q {

		typedef const Action& tAction;
		typedef const State& tState;

		enum ActionType {

			eUp,
			eDown,
			eLeft,
			eRight
		};

		struct State {

			int id;
			int x;
			int y;
			bool terminal;
		};

		struct Action {

			ActionType action;

			std::string ToString(){
				switch (action) {
				case eUp: return "Move Up";
				case eDown: return "Move Down";
				case eLeft: return "Move Left";
				case eRight: return "Move Right";
				default: return "Unknown type";
				}
			}
		};

		struct ActionStateValue {

		public:

			void InitZeros(int numStates) {
				for (int i = 0; i < numStates; ++i) {
					std::vector<float> row(4, 0);
					_qsa.push_back(row);
				}
			}

			float& operator()(tState st, tAction act){
				return get_value(st, act);
			}

		private:

			float& get_value(State& st, Action& act) {
				
				float& value = (_qsa.at(st.id)).at(act.action);
				return value;
			}

			// value table, assume memory is enough to store all of the necessary values
			std::vector<std::vector<float>> _qsa;
		};
		
		typedef ActionStateValue ac_st_value;

		class Policy {

		public:

			Policy(const ac_st_value& values):
			_values(values) {}

		private:

			const ac_st_value& _values;
		};

	}
}


#endif // __RL_Q_UTILS__
