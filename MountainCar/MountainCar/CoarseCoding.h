#ifndef __COARSE_CODING__
#define __COARSE_CODING__


/*
	This class is responsible for discretize 
	the 2D continuous state space with the method
	coarse coding.
*/

struct CoarseCoding {

public:

	CoarseCoding(
		double max_x,     // The maximum possible deviation horizontally.
		double max_xdot,  // The maximum possible horizontal speed of the car.
		int    r_slices,  // The number of slices in radial direction.
		int    phi_slices // The number of slices in central angle.
	);

	void UpdateFeatureVectorAt(double x, double x_dot);

	int GetFeatureVector();


private:

	void rescale_x(double& x);
	void rescale_xdot(double& x_dot);
	int calculate_cell_idx(double x, double x_dot);

	int calculate_phi_idx(double x, double x_dot);

private:

	// The feature vector is sparse. Contains only one value
	// which differs from zero. The value is always one. Only its 
	// position in the feature vector is enough.
	int feature_; 

	double max_x_;
	double max_xdot_;

	int r_slices_;
	int phi_slices_;
};

#endif // __COARSE_CODING__
