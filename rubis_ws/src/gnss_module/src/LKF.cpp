#include <LKF.h>

#define M_PI 3.14159265358979323846

LKF::LKF(){
    H_k = Eigen::Matrix6f::Identity();

    Q_k = Eigen::Matrix6f::Identity();

    R_k = Eigen::Matrix6f::Identity();

    x_hat_k_prev = Eigen::Vector6f::Zero();

    P_k_prev = Eigen::Matrix6f::Identity();
}

LKF::LKF(Eigen::Matrix6f& _H_k, Eigen::Matrix6f& _Q_k, Eigen::Matrix6f& _R_k, Eigen::Matrix6f& _P_k_prev){
    H_k = Eigen::Matrix6f(_H_k);
    Q_k = Eigen::Matrix6f(_Q_k);
    R_k = Eigen::Matrix6f(_R_k);
    x_hat_k_prev << 0.0f,    0.0f,      0.0f,      0.0f,      0.0f,      0.0f;
    P_k_prev = Eigen::Matrix6f(_P_k_prev);

    return;
}

void LKF::set_init_value(float init_pose_x, float init_pose_y, float init_yaw, float init_vel_x, float init_vel_y, float init_vel_yaw){
    x_hat_k_prev(0) = init_pose_x;
    x_hat_k_prev(1) = init_pose_y;
    x_hat_k_prev(2) = init_yaw;
    x_hat_k_prev(3) = init_vel_x;
    x_hat_k_prev(4) = init_vel_y;
    x_hat_k_prev(5) = init_vel_yaw;
}

Eigen::Vector6f LKF::run(float delta_t, Eigen::Vector6f& u_k, Eigen::Vector6f& z_k){ // u_k: control vector, z_k: observation vector
    // std::cout<<"prev : "<<x_hat_k_prev(0)<<" "<<x_hat_k_prev(1)<<" "<<x_hat_k_prev(2)<<std::endl;
    // Prediction
    Eigen::Matrix6f F_k = Eigen::Matrix6f::Identity(); // Prediction Matrix
    F_k(0,3) = delta_t;
    F_k(1,4) = delta_t;
    F_k(2,5) = delta_t;     

    Eigen::Matrix6f B_k = Eigen::Matrix6f::Identity() * delta_t; // Control Matrix
    B_k(0,0) = B_k(0,0) * 0.5 * delta_t;
    B_k(1,1) = B_k(1,1) * 0.5 * delta_t;
    B_k(2,2) = B_k(2,2) * 0.5 * delta_t;

    Eigen::Vector6f x_hat_k; // Predict Result
    x_hat_k = F_k * x_hat_k_prev + B_k * u_k;
    x_hat_k(2) = NormalizeRadian(x_hat_k(2), -1 * M_PI, M_PI);

    Eigen::Matrix6f P_k; // Prediction Covariance
    P_k = F_k * P_k_prev * F_k.transpose() + Q_k;

    ////////////////////////////////////////////////
    
    // Eigen::Vector6f x_hat_k; // Predict Result
    // float vel, acc, acc_yaw; 
    // vel = u_k(0); acc = u_k(1); acc_yaw = u_k(2);

    // /* yaw */
    // x_hat_k(2) = x_hat_k_prev(2) + x_hat_k_prev(5) * delta_t + 0.5 * acc_yaw * pow(delta_t,2); 
    // x_hat_k(2) = NormalizeRadian(x_hat_k(2), -1 * M_PI, M_PI);

    // /* angular velocity */
    // x_hat_k(5) = x_hat_k_prev(5) + acc_yaw * delta_t; 

    // /* pose x, y */
    // x_hat_k(0) = x_hat_k_prev(0) + vel * cos(x_hat_k_prev(2)) * delta_t + 0.5 * acc * cos(x_hat_k(2)) * pow(delta_t,2);
    // x_hat_k(1) = x_hat_k_prev(1) + vel * sin(x_hat_k_prev(2)) * delta_t + 0.5 * acc * sin(x_hat_k(2)) * pow(delta_t,2);

    // /* vel x, y */
    // x_hat_k(3) = x_hat_k_prev(3) + acc * cos(x_hat_k(2)) * delta_t;
    // x_hat_k(4) = x_hat_k_prev(4) + acc * sin(x_hat_k(2)) * delta_t;

    // Eigen::Matrix6f P_k; // Prediction Covariance
    // P_k = F_k * P_k_prev * F_k.transpose() + Q_k;


    // Update
    Eigen::Matrix6f K_prime; // Kalman gain(Modified)
    K_prime = P_k * H_k * (H_k * P_k * H_k.transpose() + R_k).completeOrthogonalDecomposition().pseudoInverse();

    Eigen::Vector6f x_hat_prime_k; // Update result
    x_hat_prime_k = x_hat_k + K_prime * (z_k - H_k * x_hat_k);

    Eigen::Matrix6f P_prime_k; // Update prediction covariance
    P_prime_k = P_k - K_prime * H_k * P_k;

    x_hat_k_prev = Eigen::Vector6f(x_hat_prime_k);
    P_k_prev = Eigen::Matrix6f(P_prime_k);

    // std::cout<<"prediction : "<<x_hat_k(0)<<" "<<x_hat_k(1)<<" "<<x_hat_k(2)<<std::endl;
    // std::cout<<"observation : "<<z_k(0)<<" "<<z_k(1)<<" "<<z_k(2)<<std::endl;
    // std::cout<<"result : "<<x_hat_prime_k(0)<<" "<<x_hat_prime_k(1)<<" "<<x_hat_prime_k(2)<<std::endl;
    // std::cout<<"####"<<std::endl;

    return x_hat_prime_k;
}