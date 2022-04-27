#include <Eigen/Core>
#include <Eigen/Dense>

class LKF{
public:
    LKF();
    LKF(float init_pose_x, float init_pose_y);
    LKF(Eigen::Matrix2f _H_k, Eigen::Matrix2f _Q_k, Eigen::Matrix2f _R_k, Eigen::Matrix2f _P_k_prev);
    void set_init_pose(float init_pose_x, float init_pose_y);
    Eigen::Vector2f run(float theta_t, Eigen::Vector2f u_k, Eigen::Vector2f z_k);
    Eigen::Vector2f run_without_update(float theta_t, Eigen::Vector2f u_k);
    void restore();
    // u_k: control vector, z_k: observation vector
private:
    Eigen::Matrix2f H_k; // Scale Matrix
    Eigen::Matrix2f Q_k; // Prediction Noise
    Eigen::Matrix2f R_k; // Observation Noise
    Eigen::Vector2f x_hat_k_prev; // Previous Value
    Eigen::Matrix2f P_k_prev; // Previous Prediction Covariance Matrix
    Eigen::Vector2f x_hat_k_prev_prev;
    Eigen::Matrix2f P_k_prev_prev;
};