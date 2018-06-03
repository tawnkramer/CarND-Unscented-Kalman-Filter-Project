#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  //state dimenion
  n_x_ = 5;
  
  //augment dimention
  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
 
  //define spreading parameter
  lambda_ = 3 - n_aug_;

  int dim = 2 * n_aug_ + 1;

  //set vector for weights
  weights_ = VectorXd(dim);
  
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  
  weights_(0) = weight_0;

  //2n+1 weights
  for (int i=1; i< dim; i++) 
  {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  Xsig_pred_ = MatrixXd(n_x_, dim);

  //Init X
  x_.setZero();

  //set example covariance matrix
  P_.setIdentity();

  //create sigma point matrix
  Xsig_aug_ = MatrixXd(n_aug_, dim);
  Xsig_aug_.setZero();

  //create sigma point matrix
  Xsig_pred_ = MatrixXd(n_aug_, dim);
  Xsig_pred_.setZero();
}

UKF::~UKF() {}

/**
 * angle normalization
 */
double NormalizeAngle(double theta)
{
    while (theta > M_PI) 
      theta -= 2. * M_PI;

    while (theta < -M_PI) 
      theta += 2. * M_PI;

    return theta;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "UKF: " << endl;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
        const float rho = meas_package.raw_measurements_[0];
        const float phi = NormalizeAngle(meas_package.raw_measurements_[1]);
        x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      	x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    previous_timestamp_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  //compute the time elapsed between the current and previous measurements
	double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = meas_package.timestamp_;

  // prevent nan
	if (abs(delta_t) < 0.000001) 
    return;

  //referencing this conversation:
  //https://discussions.udacity.com/t/numerical-instability-of-the-implementation/230449
  //Implement a strategy of limiting the larges single dt we can process
  const double dt_max = 0.05;
  
  //cout << "delta_t" << delta_t << std::endl;

  //handle larger time intervals in steps
  while (delta_t > dt_max)
  {
      //cout << "large update broken up" << std::endl;

    	Prediction(dt_max);
  
    	delta_t -= dt_max;
  }

  //process remaining dt
  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
  else
  {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) 
{

  //Create augmented sigma points matrix
  AugmentedSigmaPoints(&Xsig_aug_);

  //Use aug sig points to create sig prediction matrix
  SigmaPointPrediction(&Xsig_pred_, delta_t);

  //Use sig pred matrix to update state mean and covariance matrix
  UpdatePredState();

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  //create matrix for sigma points in measurement space
  int n_z = 2;
  int sigma_dim = 2 * n_aug_ + 1;
  MatrixXd Zsig = MatrixXd(n_z, sigma_dim);

  //transform sigma points into measurement space
  for (int i = 0; i < sigma_dim; i++) 
  {
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    // measurement model
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < sigma_dim; i++) 
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;
  S = S + R;
  

  // create a new z vector with the new measurement
  VectorXd z = VectorXd(n_z);
  
  double px = meas_package.raw_measurements_[0];
  double py = meas_package.raw_measurements_[1];

  z << px,py;
  
   //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < sigma_dim; i++)
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  
  ///* the current NIS for laser is 5.991
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) 
{

  //create matrix for sigma points in measurement space
  int n_z = 3;
  int sigma_dim = 2 * n_aug_ + 1;
  MatrixXd Zsig = MatrixXd(n_z, sigma_dim);

  //transform sigma points into measurement space
  for (int i = 0; i < sigma_dim; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = NormalizeAngle(Xsig_pred_(3,i));

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0,i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1,i) = atan2(p_y, p_x);
    Zsig(2,i) = (p_x * v1 + p_y * v2 ) / sqrt(p_x * p_x + p_y * p_y);
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) 
  {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < sigma_dim; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;

  S = S + R;
  
  // create a new z vector with the new measurement
  VectorXd z = VectorXd(n_z);
  double rho = meas_package.raw_measurements_[0];
  double phi = meas_package.raw_measurements_[1];
  double rate = meas_package.raw_measurements_[2];
  z << rho,phi,rate;
  
   //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < sigma_dim; i++) 
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  z_diff(1) = NormalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  
  ///* the current NIS for radar is 7.815
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}


void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) 
{
  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //set first column of sigma point matrix
  Xsig_out->col(0)  = x_;

  //set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig_out->col(i + 1)        = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig_out->col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }

  //print result
  //std::cout << "Xsig_out = " << std::endl << Xsig_out << std::endl;
}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) 
{
   //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;

  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1)           = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_)  = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
  
  //print result
  //std::cout << "Xsig_aug = " << std::endl << Xsig_aug_ << std::endl;
  
  //write result
  *Xsig_out = Xsig_aug;

}


/*
 * predict sigma points
 */
void UKF::SigmaPointPrediction(MatrixXd* Xsig_out, double delta_t) 
{
  int sigma_dim = 2 * n_aug_ +  1;

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, sigma_dim);

  for (int i = 0; i < sigma_dim; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) 
    {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else 
    {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }
  
  //print result
  //std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  *Xsig_out = Xsig_pred;
}


void UKF::UpdatePredState()
{  
  int dim = 2 * n_aug_ + 1;

  //predicted state mean
  x_.fill(0.0);

  for (int i = 0; i < dim; i++) 
  {
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);

  for (int i = 0; i < dim; i++) 
  {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
  
    //angle normalization
    while (x_diff(3)> M_PI) 
      x_diff(3) -= 2 * M_PI;

    while (x_diff(3)<-M_PI)
      x_diff(3) += 2 * M_PI;
  
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

