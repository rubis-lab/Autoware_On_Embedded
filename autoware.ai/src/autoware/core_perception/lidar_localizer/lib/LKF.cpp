#include <LKF.hpp>

LKF::LKF(){
    H_k <<      1.0f,       0.0f,
                0.0f,       1.0f;

    Q_k <<      1.0f,       0.0f,
                0.0f,       1.0f;

    R_k <<      1.0f,       0.0f,
                0.0f,       1.0f;

    x_hat_k_prev << 0.0f,    0.0f;

    P_k_prev << 1.0f,       0.0f,
                0.0f,       1.0f;

    x_hat_k_prev_prev << 0.0f,    0.0f;

    P_k_prev_prev <<    1.0f,       0.0f,
                        0.0f,       1.0f;
}

LKF::LKF(float init_pose_x, float init_pose_y){
    H_k <<      1.0f,       0.0f,
                0.0f,       1.0f;

    Q_k <<      1.0f,       0.0f,
                0.0f,       1.0f;

    R_k <<      1.0f,       0.0f,
                0.0f,       1.0f;

    x_hat_k_prev << init_pose_x,    init_pose_y;

    P_k_prev << 1.0f,       0.0f,
                0.0f,       1.0f;

    x_hat_k_prev_prev << init_pose_x,    init_pose_y;

    P_k_prev_prev <<    1.0f,       0.0f,
                        0.0f,       1.0f;
}

LKF::LKF(Eigen::Matrix2f _H_k, Eigen::Matrix2f _Q_k, Eigen::Matrix2f _R_k, Eigen::Matrix2f _P_k_prev){
    H_k = Eigen::Matrix2f(_H_k);
    Q_k = Eigen::Matrix2f(_Q_k);
    R_k = Eigen::Matrix2f(_R_k);
    x_hat_k_prev << 0.0f,    0.0f;
    P_k_prev = Eigen::Matrix2f(_P_k_prev);

    return;
}

void LKF::set_init_pose(float init_pose_x, float init_pose_y){
    x_hat_k_prev_prev(0) = init_pose_x;
    x_hat_k_prev_prev(1) = init_pose_y;
    x_hat_k_prev(0) = init_pose_x;
    x_hat_k_prev(1) = init_pose_y;
}

Eigen::Vector2f LKF::run(float theta_t, Eigen::Vector2f u_k, Eigen::Vector2f z_k){ // z_k: observation vector
    // Prediction
    Eigen::Matrix2f F_k; // Prediction Matrix
    F_k <<      1.0f,       0.0f,
                0.0f,       1.0f;

    Eigen::Matrix2f B_k; // Control Matrix
    B_k <<      theta_t,    0.0f,
                0.0,        theta_t;

    Eigen::Vector2f x_hat_k; // Predict Result
    x_hat_k = F_k * x_hat_k_prev + B_k * u_k;

    Eigen::Matrix2f P_k; // Prediction Covariance
    P_k = F_k * P_k_prev * F_k.transpose() + Q_k;

    // Update
    Eigen::Matrix2f K_prime; // Kalman gain(Modified)
    K_prime = P_k * H_k * (H_k * P_k * H_k.transpose() + R_k).completeOrthogonalDecomposition().pseudoInverse();

    Eigen::Vector2f x_hat_prime_k; // Update result
    x_hat_prime_k = x_hat_k + K_prime * (z_k - H_k * x_hat_k);

    Eigen::Matrix2f P_prime_k; // Update prediction covariance
    P_prime_k = P_k - K_prime * H_k * P_k;

    x_hat_k_prev_prev = Eigen::Vector2f(x_hat_k_prev);
    P_k_prev_prev = Eigen::Matrix2f(P_k_prev);

    x_hat_k_prev = Eigen::Vector2f(x_hat_prime_k);
    P_k_prev = Eigen::Matrix2f(P_prime_k);

    return x_hat_prime_k;
}

Eigen::Vector2f LKF::run_without_update(float theta_t, Eigen::Vector2f u_k){
    // Prediction
    Eigen::Matrix2f F_k; // Prediction Matrix
    F_k <<      1.0f,       0.0f,
                0.0f,       1.0f;

    Eigen::Matrix2f B_k; // Control Matrix
    B_k <<      theta_t,    0.0f,
                0.0,        theta_t;

    Eigen::Vector2f x_hat_k; // Predict Result
    x_hat_k = F_k * x_hat_k_prev + B_k * u_k;

    Eigen::Matrix2f P_k; // Prediction Covariance
    P_k = F_k * P_k_prev * F_k.transpose() + Q_k;

    // Update
    Eigen::Matrix2f K_prime; // Kalman gain(Modified)
    K_prime = P_k * H_k * (H_k * P_k * H_k.transpose() + R_k).completeOrthogonalDecomposition().pseudoInverse();

    Eigen::Vector2f z_k = x_hat_k;

    Eigen::Vector2f x_hat_prime_k; // Update result
    x_hat_prime_k = x_hat_k + K_prime * (z_k - H_k * x_hat_k);

    Eigen::Matrix2f P_prime_k; // Update prediction covariance
    P_prime_k = P_k - K_prime * H_k * P_k;

    x_hat_k_prev_prev = Eigen::Vector2f(x_hat_k_prev);
    P_k_prev_prev = Eigen::Matrix2f(P_k_prev);

    x_hat_k_prev = Eigen::Vector2f(x_hat_prime_k);
    P_k_prev = Eigen::Matrix2f(P_prime_k);

    return x_hat_prime_k;
}


void LKF::restore(){
    x_hat_k_prev = Eigen::Vector2f(x_hat_k_prev_prev);
    P_k_prev = Eigen::Matrix2f(P_k_prev_prev);

    return;  
}