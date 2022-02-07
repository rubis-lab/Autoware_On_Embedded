#include <gnss_converter/gnss_converter.h>
#include <gnss_converter/LLH2UTM.h>
#include <gnss_converter/quaternion_euler.h>

void gps_ndt_data_cb(const inertiallabs_msgs::gps_data::ConstPtr &msg_gps, const inertiallabs_msgs::ins_data::ConstPtr &msg_ins,
                     const geometry_msgs::PoseStamped::ConstPtr &msg_ndt_pose)
{

    geometry_msgs::PoseStamped cur_pose;

    double ndt_yaw, ndt_pitch, ndt_roll;

    LLH2UTM(msg_gps->LLH.x, msg_gps->LLH.y, msg_gps->LLH.z, cur_pose);

    ToEulerAngles(msg_ndt_pose->pose.orientation, ndt_yaw, ndt_pitch, ndt_roll);

    gps_stat tmp;
    geometry_msgs::Vector3 vec_tmp;

    tmp.header = msg_gps->header;

    // gps position & orientation data
    tmp.gps_pose = cur_pose.pose.position;
    vec_tmp.x = ((msg_ins->YPR.x > 180) ? (msg_ins->YPR.x - 360) : (msg_ins->YPR.x)) / 180 * M_PI;
    vec_tmp.y = ((msg_ins->YPR.y > 180) ? (msg_ins->YPR.y - 360) : (msg_ins->YPR.y)) / 180 * M_PI;
    vec_tmp.z = ((msg_ins->YPR.z > 180) ? (msg_ins->YPR.z - 360) : (msg_ins->YPR.z)) / 180 * M_PI;
    tmp.gps_ypr = vec_tmp;

    // ndt position & orientation data
    tmp.ndt_pose = msg_ndt_pose->pose.position;
    vec_tmp.x = ndt_yaw;
    vec_tmp.y = ndt_pitch;
    vec_tmp.z = ndt_roll;
    tmp.ndt_ypr = vec_tmp;

    if (ndt_pose_x_max_ < msg_ndt_pose->pose.position.x)
        ndt_pose_x_max_ = msg_ndt_pose->pose.position.x;

    if (ndt_pose_x_min_ > msg_ndt_pose->pose.position.x)
        ndt_pose_x_min_ = msg_ndt_pose->pose.position.x;

    if (ndt_pose_y_max_ < msg_ndt_pose->pose.position.y)
        ndt_pose_y_max_ = msg_ndt_pose->pose.position.y;

    if (ndt_pose_y_min_ > msg_ndt_pose->pose.position.y)
        ndt_pose_y_min_ = msg_ndt_pose->pose.position.y;

    gps_backup_.push_back(tmp);
}

void pub_gnss_pose_cb(const inertiallabs_msgs::gps_data::ConstPtr &msg_gps, const inertiallabs_msgs::ins_data::ConstPtr &msg_ins)
{
    geometry_msgs::PoseStamped cur_pose;

    cur_pose.header = msg_ins->header;
    cur_pose.header.frame_id = "/map";

    Matrix<double, 4, 1> gps;
    Matrix<double, 4, 1> ndt;

    /*================= position calculation =================*/
    LLH2UTM(msg_gps->LLH.x, msg_gps->LLH.y, msg_gps->LLH.z, cur_pose);

    gps(0, 0) = cur_pose.pose.position.x;
    gps(1, 0) = cur_pose.pose.position.y;
    gps(2, 0) = cur_pose.pose.position.z;
    gps(3, 0) = 1.0;

    ndt = pos_tf_ * gps;

    cur_pose.pose.position.x = ndt(0, 0);
    cur_pose.pose.position.y = ndt(1, 0);
    cur_pose.pose.position.z = ndt(2, 0);
    /*=======================================================*/

    /*=============== orientation calculation ===============*/
    gps(0, 0) = ((msg_ins->YPR.x > 180) ? (msg_ins->YPR.x - 360) : (msg_ins->YPR.x)) / 180 * M_PI;
    gps(1, 0) = ((msg_ins->YPR.y > 180) ? (msg_ins->YPR.y - 360) : (msg_ins->YPR.y)) / 180 * M_PI;
    gps(2, 0) = ((msg_ins->YPR.z > 180) ? (msg_ins->YPR.z - 360) : (msg_ins->YPR.z)) / 180 * M_PI;
    gps(3, 0) = 1.0;

    ndt = ori_tf_ * gps;

    ToQuaternion(ndt(0, 0), ndt(1, 0), ndt(2, 0), cur_pose.pose.orientation);

    gnss_pose_pub_.publish(cur_pose);
    /*=======================================================*/
}

void scale_image(int pos, void *userdata)
{
    scale_factor_ = ((pos < 1) ? (1) : (pos));
}

void mouse_cb(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_RBUTTONDOWN)
    {
        double point_x = ndt_pose_x_min_ + x / scale_factor_;
        double point_y = ndt_pose_y_min_ + y / scale_factor_;
        double min_dist = 9999;
        int idx;

        for (int i = 0; i < gps_backup_.size(); i++)
        {
            if (pow((point_x - gps_backup_[i].ndt_pose.x), 2) + pow((point_y - gps_backup_[i].ndt_pose.y), 2) < pow(min_dist, 2))
            {
                idx = i;
                min_dist = pow((point_x - gps_backup_[i].ndt_pose.x), 2) + pow((point_y - gps_backup_[i].ndt_pose.y), 2);
            }
        }

        selected_points_[points_idx_] = gps_backup_[idx];
        std::cout << "**************** Point INFO ****************" << std::endl;

        std::cout << "GPS Pose" << std::endl;
        std::cout << "    x : " << selected_points_[points_idx_].gps_pose.x << std::endl;
        std::cout << "    y : " << selected_points_[points_idx_].gps_pose.y << std::endl;
        std::cout << "    z : " << selected_points_[points_idx_].gps_pose.z << std::endl;

        std::cout << "GPS YPR" << std::endl;
        std::cout << "    yaw : " << selected_points_[points_idx_].gps_ypr.x / M_PI * 180 << std::endl;
        std::cout << "    pitch : " << selected_points_[points_idx_].gps_ypr.y / M_PI * 180 << std::endl;
        std::cout << "    roll : " << selected_points_[points_idx_].gps_ypr.z / M_PI * 180 << std::endl;

        std::cout << "NDT Pose" << std::endl;
        std::cout << "    x : " << selected_points_[points_idx_].ndt_pose.x << std::endl;
        std::cout << "    y : " << selected_points_[points_idx_].ndt_pose.y << std::endl;
        std::cout << "    z : " << selected_points_[points_idx_].ndt_pose.z << std::endl;

        std::cout << "NDT YPR" << std::endl;
        std::cout << "    yaw : " << selected_points_[points_idx_].ndt_ypr.x / M_PI * 180 << std::endl;
        std::cout << "    pitch : " << selected_points_[points_idx_].ndt_ypr.y / M_PI * 180 << std::endl;
        std::cout << "    roll : " << selected_points_[points_idx_].ndt_ypr.z / M_PI * 180 << std::endl;

        std::cout << "NDT score" << selected_points_[points_idx_].ndt_score << std::endl;
    }
}

void points_select()
{
    cv::Mat orig_img = cv::Mat((ndt_pose_y_max_ - ndt_pose_y_min_ + 1), (ndt_pose_x_max_ - ndt_pose_x_min_ + 1), CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat display_img;

    for (int i = 0; i < gps_backup_.size(); i++)
    {
        cv::circle(orig_img, cv::Point((gps_backup_[i].ndt_pose.x - ndt_pose_x_min_), (gps_backup_[i].ndt_pose.y - ndt_pose_y_min_)),
                   1, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }

    orig_img.copyTo(display_img);

    cv::namedWindow("Display Image");
    cv::createTrackbar("Scale", "Display Image", &scale_factor_, 100, scale_image, NULL);

    cv::setMouseCallback("Display Image", mouse_cb, NULL);

    std::cout << "**************** How To Use ****************" << std::endl;
    std::cout << "If you want to end, press the ESC button." << std::endl;
    std::cout << "If you want to choose a point, press the number and click the point in the image." << std::endl;
    std::cout << "To change the image size, use track bar to select a value and press enter." << std::endl;
    std::cout << "********************************************" << std::endl;

    int keycode;
    while (true)
    {
        display_img = cv::Mat(orig_img.rows * scale_factor_, orig_img.cols * scale_factor_, CV_8UC3, cv::Scalar(255, 255, 255));
        for (int i = 0; i < gps_backup_.size(); i++)
        {
            cv::circle(display_img,
                       cv::Point(scale_factor_ * (gps_backup_[i].ndt_pose.x - ndt_pose_x_min_), scale_factor_ * (gps_backup_[i].ndt_pose.y - ndt_pose_y_min_)),
                       1, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }

        cv::imshow("Display Image", display_img);
        keycode = cv::waitKey();

        if (keycode == ESC_BUTTON)
            break;

        switch (keycode)
        {
        case '1':
            points_idx_ = 0;
            std::cout << "Point Number : 1" << std::endl;
            break;
        case '2':
            points_idx_ = 1;
            std::cout << "Point Number : 2" << std::endl;
            break;
        case '3':
            points_idx_ = 2;
            std::cout << "Point Number : 3" << std::endl;
            break;
        case '4':
            points_idx_ = 3;
            std::cout << "Point Number : 4" << std::endl;
            break;
        case ENTER_BUTTON:
            break;
        default:
            std::cout << "Unknown keyboard input" << std::endl;
            break;
        }
    }
}

void calculate_tf_matrix()
{
    Matrix<double, 4, 4> gps_pos, ndt_pos, gps_ypr, ndt_ypr;

    for (int i = 0; i < 4; i++)
    {
        gps_pos(0, i) = selected_points_[i].gps_pose.x;
        gps_pos(1, i) = selected_points_[i].gps_pose.y;
        gps_pos(2, i) = selected_points_[i].gps_pose.z;
        gps_pos(3, i) = 1.0;

        ndt_pos(0, i) = selected_points_[i].ndt_pose.x;
        ndt_pos(1, i) = selected_points_[i].ndt_pose.y;
        ndt_pos(2, i) = selected_points_[i].ndt_pose.z;
        ndt_pos(3, i) = 1.0;

        gps_ypr(0, i) = selected_points_[i].gps_ypr.x;
        gps_ypr(1, i) = selected_points_[i].gps_ypr.y;
        gps_ypr(2, i) = selected_points_[i].gps_ypr.z;
        gps_ypr(3, i) = 1.0;

        ndt_ypr(0, i) = selected_points_[i].ndt_ypr.x;
        ndt_ypr(1, i) = selected_points_[i].ndt_ypr.y;
        ndt_ypr(2, i) = selected_points_[i].ndt_ypr.z;
        ndt_ypr(3, i) = 1.0;
    }

    pos_tf_ = ndt_pos * gps_pos.inverse();
    ori_tf_ = ndt_ypr * gps_ypr.inverse();

    printf("========== position tf matrix ==========\n");
    printf("[%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf]\n", pos_tf_(0, 0), pos_tf_(0, 1), pos_tf_(0, 2), pos_tf_(0, 3), pos_tf_(1, 0), pos_tf_(1, 1), pos_tf_(1, 2), pos_tf_(1, 3), pos_tf_(2, 0), pos_tf_(2, 1), pos_tf_(2, 2), pos_tf_(2, 3), pos_tf_(3, 0), pos_tf_(3, 1), pos_tf_(3, 2), pos_tf_(3, 3));
    printf("======== orientation tf matrix =========\n");
    printf("[%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf]\n", ori_tf_(0, 0), ori_tf_(0, 1), ori_tf_(0, 2), ori_tf_(0, 3), ori_tf_(1, 0), ori_tf_(1, 1), ori_tf_(1, 2), ori_tf_(1, 3), ori_tf_(2, 0), ori_tf_(2, 1), ori_tf_(2, 2), ori_tf_(2, 3), ori_tf_(3, 0), ori_tf_(3, 1), ori_tf_(3, 2), ori_tf_(3, 3));
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "gnss_converter");

    ros::NodeHandle nh;

    bool calculate_tf;

    gnss_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/gnss_pose", 10);

    message_filters::Subscriber<inertiallabs_msgs::gps_data> gps_sub(nh, "/Inertial_Labs/gps_data", 10);
    message_filters::Subscriber<geometry_msgs::PoseStamped> ndt_pose_sub(nh, "/ndt_pose", 10);
    message_filters::Subscriber<inertiallabs_msgs::ins_data> ins_sub(nh, "/Inertial_Labs/ins_data", 10);

    Synchronizer<SyncPolicy_1> sync_1(SyncPolicy_1(100), gps_sub, ins_sub, ndt_pose_sub);
    Synchronizer<SyncPolicy_2> sync_2(SyncPolicy_2(10), gps_sub, ins_sub);

    ros::param::get("/gnss_converter/calculate_tf", calculate_tf);

    if (calculate_tf)
    {
        sync_1.registerCallback(boost::bind(&gps_ndt_data_cb, _1, _2, _3));

        pid_t pid;
        if ((pid = fork()) < 0)
            ROS_ERROR("Cannot create child!");

        if (pid == 0)
        {
            std::string file_path;
            char file_path_cstr[150];

            ros::param::get("/gnss_converter/bag_file_path", file_path);

            int str_len = file_path.length();
            if (str_len >= 150)
            {
                ROS_ERROR("File path is too long!!");
                exit(0);
            }

            strcpy(file_path_cstr, file_path.c_str());

            const char *file = "/opt/ros/melodic/bin/rosbag";
            char *exe_argv[] = {"/opt/ros/melodic/bin/rosbag",
                                "play",
                                "-r",
                                "5",
                                file_path_cstr,
                                NULL};

            if (execvp(file, exe_argv) < 0)
            {
                ROS_ERROR("Cannot load bag file!!");
            }
        }
        else
        {
            int wstatus;
            while (waitpid(pid, &wstatus, WNOHANG) == 0)
            {
                ros::spinOnce();
            }
            ROS_WARN("Finish loading data from rosbag file!!");
            points_select();
            calculate_tf_matrix();
        }
    }

    else
    {
        sync_2.registerCallback(boost::bind(&pub_gnss_pose_cb, _1, _2));

        vector<double> tf_tmp;

        /*================= pos_tf_ matrix =================*/
        ros::param::get("/gnss_converter/pos_tf", tf_tmp);
        pos_tf_(0, 0) = tf_tmp[0];
        pos_tf_(0, 1) = tf_tmp[1];
        pos_tf_(0, 2) = tf_tmp[2];
        pos_tf_(0, 3) = tf_tmp[3];
        pos_tf_(1, 0) = tf_tmp[4];
        pos_tf_(1, 1) = tf_tmp[5];
        pos_tf_(1, 2) = tf_tmp[6];
        pos_tf_(1, 3) = tf_tmp[7];
        pos_tf_(2, 0) = tf_tmp[8];
        pos_tf_(2, 1) = tf_tmp[9];
        pos_tf_(2, 2) = tf_tmp[10];
        pos_tf_(2, 3) = tf_tmp[11];
        pos_tf_(3, 0) = tf_tmp[12];
        pos_tf_(3, 1) = tf_tmp[13];
        pos_tf_(3, 2) = tf_tmp[14];
        pos_tf_(3, 3) = tf_tmp[15];
        /*=================================================*/

        /*================= ori_tf_ matrix =================*/
        ros::param::get("/gnss_converter/ori_tf", tf_tmp);
        ori_tf_(0, 0) = tf_tmp[0];
        ori_tf_(0, 1) = tf_tmp[1];
        ori_tf_(0, 2) = tf_tmp[2];
        ori_tf_(0, 3) = tf_tmp[3];
        ori_tf_(1, 0) = tf_tmp[4];
        ori_tf_(1, 1) = tf_tmp[5];
        ori_tf_(1, 2) = tf_tmp[6];
        ori_tf_(1, 3) = tf_tmp[7];
        ori_tf_(2, 0) = tf_tmp[8];
        ori_tf_(2, 1) = tf_tmp[9];
        ori_tf_(2, 2) = tf_tmp[10];
        ori_tf_(2, 3) = tf_tmp[11];
        ori_tf_(3, 0) = tf_tmp[12];
        ori_tf_(3, 1) = tf_tmp[13];
        ori_tf_(3, 2) = tf_tmp[14];
        ori_tf_(3, 3) = tf_tmp[15];
        /*================================================*/

        ros::spin();
    }

    return 0;
}