#include <gnss_converter/gnss_converter.h>

void calculate_tf_with_gps_ndt_cb(const inertiallabs_msgs::gps_data::ConstPtr& msg_gps, const inertiallabs_msgs::ins_data::ConstPtr& msg_ins,
                                    const geometry_msgs::PoseStamped::ConstPtr& msg_ndt_pose){
    geometry_msgs::PoseStamped cur_pose;
    double ndt_yaw, ndt_pitch, ndt_roll;

    LLH2UTM(msg_gps->LLH.x, msg_gps->LLH.y, msg_gps->LLH.z, cur_pose);

    ToEulerAngles(msg_ndt_pose->pose.orientation, ndt_yaw, ndt_pitch, ndt_roll);

    if(cb_count % 100 == 1 && cb_count <= 301){
        gps_pos(0, cb_count / 100) = cur_pose.pose.position.x; gps_pos(1, cb_count / 100) = cur_pose.pose.position.y;
        gps_pos(2, cb_count / 100) = cur_pose.pose.position.z; gps_pos(3, cb_count / 100) = 1.0;

        ndt_pos(0, cb_count / 100) = msg_ndt_pose->pose.position.x; ndt_pos(1, cb_count / 100) = msg_ndt_pose->pose.position.y;
        ndt_pos(2, cb_count / 100) = msg_ndt_pose->pose.position.z; ndt_pos(3, cb_count / 100) = 1.0;

        gps_qt(0, cb_count / 100) = msg_ins->YPR.x / 180 * M_PI; gps_qt(1, cb_count / 100) = msg_ins->YPR.y / 180 * M_PI;
        gps_qt(2, cb_count / 100) = msg_ins->YPR.z / 180 * M_PI; gps_qt(3, cb_count / 100) = 1.0;

        ndt_qt(0, cb_count / 100) = ndt_yaw;  ndt_qt(1, cb_count / 100) = ndt_pitch;
        ndt_qt(2, cb_count / 100) = ndt_roll; ndt_qt(3, cb_count / 100) = 1.0;
    }

    if(cb_count == 301){
        pos_tf = ndt_pos * gps_pos.inverse();
        ori_tf = ndt_qt * gps_qt.inverse();

        ROS_INFO("===== position tf matrix =====");
        ROS_INFO("%10f %10f %10f %10f", pos_tf(0, 0), pos_tf(0, 1), pos_tf(0, 2), pos_tf(0, 3));
        ROS_INFO("%10f %10f %10f %10f", pos_tf(1, 0), pos_tf(1, 1), pos_tf(1, 2), pos_tf(1, 3));
        ROS_INFO("%10f %10f %10f %10f", pos_tf(2, 0), pos_tf(2, 1), pos_tf(2, 2), pos_tf(2, 3));
        ROS_INFO("%10f %10f %10f %10f", pos_tf(3, 0), pos_tf(3, 1), pos_tf(3, 2), pos_tf(3, 3));
        ROS_INFO("==============================");

        ROS_INFO("==== orientation tf matrix ====");
        ROS_INFO("%10f %10f %10f %10f", ori_tf(0, 0), ori_tf(0, 1), ori_tf(0, 2), ori_tf(0, 3));
        ROS_INFO("%10f %10f %10f %10f", ori_tf(1, 0), ori_tf(1, 1), ori_tf(1, 2), ori_tf(1, 3));
        ROS_INFO("%10f %10f %10f %10f", ori_tf(2, 0), ori_tf(2, 1), ori_tf(2, 2), ori_tf(2, 3));
        ROS_INFO("%10f %10f %10f %10f", ori_tf(3, 0), ori_tf(3, 1), ori_tf(3, 2), ori_tf(3, 3));
        ROS_INFO("===============================");

        ros::shutdown();
    }
    
    cb_count++;
}

void pub_gnss_pose_cb(const inertiallabs_msgs::gps_data::ConstPtr& msg_gps, const inertiallabs_msgs::ins_data::ConstPtr& msg_ins){
    geometry_msgs::PoseStamped cur_pose;

    cur_pose.header = msg_gps->header;
    cur_pose.header.frame_id = "/map";

    Matrix<double, 4, 1> gps;
    Matrix<double, 4, 1> ndt;
    
    // position calculation

    LLH2UTM(msg_gps->LLH.x, msg_gps->LLH.y, msg_gps->LLH.z, cur_pose);

    gps(0, 0) = cur_pose.pose.position.x; gps(1, 0) = cur_pose.pose.position.y;
    gps(2, 0) = cur_pose.pose.position.z; gps(3, 0) = 1.0;

    ndt = pos_tf * gps;

    cur_pose.pose.position.x = ndt(0, 0);
    cur_pose.pose.position.y = ndt(1, 0);
    cur_pose.pose.position.z = ndt(2, 0);

    // orientation calculation

    gps(0, 0) = msg_ins->YPR.x / 180 * M_PI; gps(1, 0) = msg_ins->YPR.y / 180 * M_PI;
    gps(2, 0) = msg_ins->YPR.z / 180 * M_PI; gps(3, 0) = 1.0;

    ndt = ori_tf * gps;

    ToQuaternion(ndt(0,0), ndt(1,0), ndt(2,0), cur_pose.pose.orientation);

    gnss_pose_pub.publish(cur_pose);
}

void LLH2UTM(double Lat, double Long, double H, geometry_msgs::PoseStamped& pose){
    double a = WGS84_A;
    double eccSquared = UTM_E2;
    double k0 = UTM_K0;
    double LongOrigin;
    double eccPrimeSquared;
    double N, T, C, A, M;
    // Make sure the longitude is between -180.00 .. 179.9
    // (JOQ: this is broken for Long < -180, do a real normalize)
    double LongTemp = (Long+180)-int((Long+180)/360)*360-180;
    double LatRad = angles::from_degrees(Lat);
    double LongRad = angles::from_degrees(LongTemp);
    double LongOriginRad;
    pose.pose.position.z = H;
    // Fix Zone number with Korea
    int zone = 52;
    char band = 'S';
    // +3 puts origin in middle of zone
    LongOrigin = (zone - 1)*6 - 180 + 3;
    LongOriginRad = angles::from_degrees(LongOrigin);
    eccPrimeSquared = (eccSquared)/(1-eccSquared);
    N = a/sqrt(1-eccSquared*sin(LatRad)*sin(LatRad));
    T = tan(LatRad)*tan(LatRad);
    C = eccPrimeSquared*cos(LatRad)*cos(LatRad);
    A = cos(LatRad)*(LongRad-LongOriginRad);
    M = a*((1 - eccSquared/4 - 3*eccSquared*eccSquared/64
        - 5*eccSquared*eccSquared*eccSquared/256) * LatRad
        - (3*eccSquared/8 + 3*eccSquared*eccSquared/32
        + 45*eccSquared*eccSquared*eccSquared/1024)*sin(2*LatRad)
        + (15*eccSquared*eccSquared/256
        + 45*eccSquared*eccSquared*eccSquared/1024)*sin(4*LatRad)
        - (35*eccSquared*eccSquared*eccSquared/3072)*sin(6*LatRad));
    pose.pose.position.y = (double)
    (k0*N*(A+(1-T+C)*A*A*A/6
        + (5-18*T+T*T+72*C-58*eccPrimeSquared)*A*A*A*A*A/120)
    + 500000.0);
    pose.pose.position.x = (double)
    (k0*(M+N*tan(LatRad)
        *(A*A/2+(5-T+9*C+4*C*C)*A*A*A*A/24
        + (61-58*T+T*T+600*C-330*eccPrimeSquared)*A*A*A*A*A*A/720)));
    
    double TM[4][4] = 
    {{-0.821456, -0.593423, -0.006448, 3606301.475406},
    {-0.596954, 0.803991, -0.096993, 2231713.639404},
    {0.049875, 0.018177, -0.047063, -213252.081285},
    {0.000000, 0.000000, 0.000000, 1.000000}};

    double input[4] = {pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, 1};
    pose.pose.position.x = TM[0][0]*input[0] + TM[0][1]*input[1] + TM[0][2]*input[2] + TM[0][3]*input[3];
    pose.pose.position.y = TM[1][0]*input[0] + TM[1][1]*input[1] + TM[1][2]*input[2] + TM[1][3]*input[3];
    pose.pose.position.z = TM[2][0]*input[0] + TM[2][1]*input[1] + TM[2][2]*input[2] + TM[2][3]*input[3];    
}

void ToEulerAngles(geometry_msgs::Quaternion q, double &yaw, double &pitch, double &roll){
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        pitch = std::asin(sinp);

    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

void ToQuaternion(double yaw, double pitch, double roll, geometry_msgs::Quaternion &q)
{
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;
}

int main(int argc, char *argv[]){

    ros::init(argc, argv, "gnss_converter");

    ros::NodeHandle nh;

    bool calculate_tf;

    gnss_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/gnss_pose", 10);

    message_filters::Subscriber<inertiallabs_msgs::gps_data> gps_sub(nh, "/Inertial_Labs/gps_data", 3);
    message_filters::Subscriber<geometry_msgs::PoseStamped> ndt_pose_sub(nh, "/ndt_pose", 3);
    message_filters::Subscriber<inertiallabs_msgs::ins_data> ins_sub(nh, "/Inertial_Labs/ins_data", 3);
    
    Synchronizer<SyncPolicy_1> sync_1(SyncPolicy_1(10), gps_sub, ins_sub, ndt_pose_sub);
    Synchronizer<SyncPolicy_2> sync_2(SyncPolicy_2(10), gps_sub, ins_sub);

    ros::param::get("/gnss_converter/calculate_tf", calculate_tf);

    if(calculate_tf){
        sync_1.registerCallback(boost::bind(&calculate_tf_with_gps_ndt_cb, _1, _2, _3));
    }

    else{
        sync_2.registerCallback(boost::bind(&pub_gnss_pose_cb, _1, _2));

        vector<double> tf_tmp;

        /*================= pos_tf matrix =================*/
        ros::param::get("/gnss_converter/pos_tf", tf_tmp);
        pos_tf(0, 0) = tf_tmp[0]; pos_tf(0, 1) = tf_tmp[1]; pos_tf(0, 2) = tf_tmp[2]; pos_tf(0, 3) = tf_tmp[3];
        pos_tf(1, 0) = tf_tmp[4]; pos_tf(1, 1) = tf_tmp[5]; pos_tf(1, 2) = tf_tmp[6]; pos_tf(1, 3) = tf_tmp[7];
        pos_tf(2, 0) = tf_tmp[8]; pos_tf(2, 1) = tf_tmp[9]; pos_tf(2, 2) = tf_tmp[10]; pos_tf(2, 3) = tf_tmp[11];
        pos_tf(3, 0) = tf_tmp[12]; pos_tf(3, 1) = tf_tmp[13]; pos_tf(3, 2) = tf_tmp[14]; pos_tf(3, 3) = tf_tmp[15];
        /*=================================================*/

        /*================= ori_tf matrix =================*/
        ros::param::get("/gnss_converter/ori_tf", tf_tmp);
        ori_tf(0, 0) = tf_tmp[0]; ori_tf(0, 1) = tf_tmp[1]; ori_tf(0, 2) = tf_tmp[2]; ori_tf(0, 3) = tf_tmp[3];
        ori_tf(1, 0) = tf_tmp[4]; ori_tf(1, 1) = tf_tmp[5]; ori_tf(1, 2) = tf_tmp[6]; ori_tf(1, 3) = tf_tmp[7];
        ori_tf(2, 0) = tf_tmp[8]; ori_tf(2, 1) = tf_tmp[9]; ori_tf(2, 2) = tf_tmp[10]; ori_tf(2, 3) = tf_tmp[11];
        ori_tf(3, 0) = tf_tmp[12]; ori_tf(3, 1) = tf_tmp[13]; ori_tf(3, 2) = tf_tmp[14]; ori_tf(3, 3) = tf_tmp[15];
        /*================================================*/
    }

    while(ros::ok()){
        ros::spin();
    }

    return 0;
}