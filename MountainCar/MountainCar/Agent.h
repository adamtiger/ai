#ifndef __AGENT__
#define __AGENT__

#include <vector>
using namespace std;


enum Action {
	LEFT,
	RIGHT
};

/*
	This is the acting agent in the environment
	which learns to drive the car out the valley.
*/

class Agent {

public:

	Agent(
		int    nm_features, // The number of the states after the continuous space was sliced. 
		double alpha,       // The learning rate.
		double gamma        // The discounting factor.
	);

	void Initialize(int state);

	void UpdateThetaValues(int state, Action action, double reward, int next_state, Action next_action);

	Action Policy();

private:

	double get_th(int state, Action action) const;
	void set_th(int state, Action action, double new_value);

private:

	vector<double> left_thetas_; // parameters for linear approximators
	vector<double> right_thetas_;

	int state_; // current state

	int nm_features_;

	double alpha_;
	double gamma_;

};

#endif // __AGENT__
