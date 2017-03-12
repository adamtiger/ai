#include "CoarseCoding.h"
#include <math.h>

#define PI 3.14

using namespace std;

CoarseCoding::CoarseCoding(double max_x, double max_xdot, int r_slices, int phi_slices){

	max_x_ = max_x;
	max_xdot_ = max_xdot;
	r_slices_ = r_slices;
	phi_slices_ = phi_slices;
}

int CoarseCoding::GetFeatureVectorAt(double x, double x_dot){

	rescale_x(x);
	rescale_xdot(x_dot);
	feature_ = calculate_cell_idx(x, x_dot);
	return feature_;
}


void CoarseCoding::rescale_x(double& x){

	x = x / max_x_;
}

void CoarseCoding::rescale_xdot(double& x_dot){

	x_dot = x_dot / max_xdot_;
}

int CoarseCoding::calculate_cell_idx(double x, double x_dot){

	int r_idx = floor(sqrt(x*x + x_dot*x_dot) * r_slices_);
	
	int phi_idx = calculate_phi_idx(x, x_dot);

	return r_idx * phi_slices_ + phi_idx;
}

int CoarseCoding::calculate_phi_idx(double x, double x_dot){

	double phi = fabs(atan(x_dot / x));

	if (fabs(x_dot) < 0.00000001) { // Use a tolerance value instead of the exact equality.
		if (fabs(x) < 0.00000001)
			phi = 0.0;
		else {
			phi = signbit(x) ? PI : 0.0;
		}
	}
	else if (fabs(x) < 0.00000001) {
		phi = PI / 2.0;
	}
	else if (x > 0.0 && x_dot < 0.0) {
		phi = 2 * PI - phi;
	}
	else if (x < 0.0 && x_dot > 0.0) {
		phi = PI - phi;
	}
	else if (x<0.0 && x_dot > 0.0) {
		phi = PI + phi;
	}

	return floor((phi / (2.0 * PI)) * (double)phi_slices_);
}
