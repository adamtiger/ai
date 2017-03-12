#ifndef __RL_Q_UTILS__
#define __RL_Q_UTILS__

#include <string>
#include <vector>

namespace rl {
	namespace q {

#define NUMBER_OF_ACTIONS 4

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

			State(int id, int x, int y, bool terminal) {
				this->id = id;
				this->x = x; this->y = y;
				this->terminal = terminal;
			}
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

			void get_deviation(int* dx_ou, int* dy_ou) const{
				switch (action) {
				case eUp:    *dx_ou = 0; *dy_ou = 1; break;
				case eDown:  *dx_ou = 0; *dy_ou = -1; break;
				case eLeft:  *dx_ou = -1; *dy_ou = 0; break;
				case eRight: *dx_ou = 1; *dy_ou = 0; break;
				}
			}
		};

		typedef const Action& tAction;
		typedef const State& tState;

		struct ActionStateValue {

		public:

			explicit ActionStateValue(int numStates) {
				InitZeros(numStates);
			}

			void InitZeros(int numStates) {
				for (int i = 0; i < numStates; ++i) {
					std::vector<float> row(NUMBER_OF_ACTIONS, 0);
					_qsa.push_back(row);
				}
			}

			float& operator()(tState st, tAction act){
				return get_value(st, act);
			}

			const std::vector<float>& operator()(tState st) const {
				return _qsa.at(st.id);
			}

		private:

			float& get_value(tState st, tAction act){
				
				float& value = (_qsa.at(st.id)).at(act.action);
				return value;
			}

			// value table, assume memory is enough to store all of the necessary values
			std::vector<std::vector<float>> _qsa;
		};
		
		typedef ActionStateValue ac_st_value;

		class Policy {

		public:

			Policy(ac_st_value& values):
			_values(values) {}

			Action use_policy(tState st) {
				Action act;
				int idx = 0;
				float value = 0.0f;
				for (int i = 0; i < NUMBER_OF_ACTIONS; ++i) {
					act.action = (ActionType)i;
					if (value < _values(st, act)) { value = _values(st, act); idx = i; }
				}

				const int prob_eps = 3; // 3 transitions have this ratio
				const int prob_best = 91;
				int prob = rand() % 100;
				int flag = 0;
				bool cont = true;
				for (int i = 0; cont && i < NUMBER_OF_ACTIONS; ++i) {
					i == idx ? flag += prob_best : flag += prob_eps;
					if (prob < flag) { idx = i; cont = false; }
				}

				act.action = (ActionType)idx;
				return act;
			}

		private:

			ac_st_value& _values;
		};

	}
}


#endif // __RL_Q_UTILS__
