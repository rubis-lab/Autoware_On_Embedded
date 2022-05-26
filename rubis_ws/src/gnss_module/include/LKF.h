#include <Eigen/Core>
#include <Eigen/Dense>
#include "quaternion_euler.h"

namespace Eigen{
typedef Eigen::Matrix<float, 6,6> Matrix6f;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
}

class LKF{
public:
    LKF();
    LKF(Eigen::Matrix6f& _H_k, Eigen::Matrix6f& _Q_k, Eigen::Matrix6f& _R_k, Eigen::Matrix6f& _P_k_prev);
    void set_init_value(float init_pose_x, float init_pose_y, float init_yaw, float init_vel_x, float init_vel_y, float init_vel_yaw);
    Eigen::Vector6f run(float delta_t, Eigen::Vector6f& u_k, Eigen::Vector6f& z_k);
    // u_k: control vector, z_k: observation vector
private:
    Eigen::Matrix6f H_k; // Scale Matrix
    Eigen::Matrix6f Q_k; // Prediction Noise
    Eigen::Matrix6f R_k; // Observation Noise
    Eigen::Vector6f x_hat_k_prev; // Previous Value
    Eigen::Matrix6f P_k_prev; // Previous Prediction Covariance Matrix
};