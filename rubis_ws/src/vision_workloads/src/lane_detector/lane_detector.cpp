#include "lane_detector.hpp"

LaneDetector::LaneDetector() {
    ros::NodeHandle private_nh("~");
    std::string node_name = ros::this_node::getName();
    std::string input_topic;
    std::string output_topic;
    output_topic = node_name + std::string("_lane");

    private_nh.param<std::string>(node_name + "/input_topic", input_topic,
                                  "/svl_image_raw");
    private_nh.param<bool>(node_name + "/debug", debug_, false);
    private_nh.param<bool>(node_name + "/filter_image", filter_image_, false);
    private_nh.param<bool>(node_name + "/gray_scale", gray_scale_, false);
    private_nh.param<bool>(node_name + "/gaussian_blur", gaussian_blur_, false);
    private_nh.param<bool>(node_name + "/canny", canny_, false);
    private_nh.param<bool>(node_name + "/roi", roi_, false);
    lane_pub_ = nh_.advertise<rubis_msgs::Bool>(output_topic.c_str(), 1);
    image_sub_ =
        nh_.subscribe(input_topic, 1, &LaneDetector::imageCallback, this);

    return;
}

void LaneDetector::run() {
    ros::NodeHandle private_nh("~");
    std::string node_name = ros::this_node::getName();
    std::string task_response_time_filename;
    private_nh.param<std::string>(
        node_name + "/task_response_time_filename", task_response_time_filename,
        "~/Documents/profiling/response_time/lane_detector.csv");

    struct rubis::sched_attr attr;
    std::string policy;
    int priority, exec_time, deadline, period;

    private_nh.param(node_name + "/task_scheduling_configs/policy", policy,
                     std::string("NONE"));
    private_nh.param(node_name + "/task_scheduling_configs/priority", priority,
                     99);
    private_nh.param(node_name + "/task_scheduling_configs/exec_time",
                     exec_time, 0);
    private_nh.param(node_name + "/task_scheduling_configs/deadline", deadline,
                     0);
    private_nh.param(node_name + "/task_scheduling_configs/period", period, 0);

    attr = rubis::create_sched_attr(priority, exec_time, deadline, period);
    rubis::init_task_scheduling(policy, attr);
    rubis::init_task_profiling(task_response_time_filename);

    while (ros::ok()) {
        ros::spinOnce();
    }

    return;
}

void LaneDetector::imageCallback(const rubis_msgs::Image &image) {
    rubis::start_task_profiling();

    rubis::instance_ = image.instance;
    rubis::vision_instance_ = image.instance;

    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image.msg, "bgr8");
    Mat frame = cv_image->image;

    // Determine if video is taken during daytime or not
    bool isDay = isDayTime(frame);

    // Filter image
    Mat filteredIMG;
    if (filter_image_)
        filteredIMG = filterColors(frame, isDay);
    else
        filteredIMG = frame;

    // Apply grayscale
    Mat gray;
    if (gray_scale_)
        gray = applyGrayscale(frame);
    else
        gray = filteredIMG;

    // Apply Gaussian blur
    Mat gBlur;
    if (gaussian_blur_)
        gBlur = applyGaussianBlur(gray);
    else
        gBlur = gray;

    // Find edges
    Mat edges;
    if (canny_)
        edges = applyCanny(gBlur);
    else
        edges = gBlur;

    // Create mask (Region of Interest)
    Mat maskedIMG;
    if (roi_)
        maskedIMG = RegionOfInterest(edges);
    else
        maskedIMG = edges;

    // Detect straight lines and draw the lanes if possible
    std::vector<Vec4i> linesP = houghLines(maskedIMG, frame.clone(), false);

    // Draw lines
    Mat lanes = drawLanes(frame, linesP);

    // cv::resize(lanes, lanes, cv::Size(640, 480));
    // imshow("Lanes", lanes);
    // waitKey(1);

    rubis_msgs::Bool lane_msg;
    lane_msg.header = image.header;
    lane_msg.instance = rubis::instance_;
    lane_pub_.publish(lane_msg);

    rubis::stop_task_profiling(rubis::instance_++, rubis::lidar_instance_++,
                               rubis::vision_instance_++);
}

Mat LaneDetector::applyGrayscale(Mat source) {
    Mat dst;
    cvtColor(source, dst, COLOR_BGR2GRAY);

    return dst;
}

Mat LaneDetector::applyGaussianBlur(Mat source) {
    Mat dst;
    GaussianBlur(source, dst, Size(3, 3), 0);

    return dst;
}

Mat LaneDetector::applyCanny(Mat source) {
    Mat dst;
    Canny(source, dst, 50, 150);

    return dst;
}

Mat LaneDetector::filterColors(Mat source, bool isDayTime) {
    Mat hsv, whiteMask, whiteImage, yellowMask, yellowImage, whiteYellow;

    // White mask
    std::vector<int> lowerWhite = {130, 130, 130};
    std::vector<int> upperWhite = {255, 255, 255};
    inRange(source, lowerWhite, upperWhite, whiteMask);
    bitwise_and(source, source, whiteImage, whiteMask);

    // Yellow mask
    cvtColor(source, hsv, COLOR_BGR2HSV);
    std::vector<int> lowerYellow = {20, 100, 110};
    std::vector<int> upperYellow = {30, 180, 240};
    inRange(hsv, lowerYellow, upperYellow, yellowMask);
    bitwise_and(source, source, yellowImage, yellowMask);

    // Blend yellow and white together
    addWeighted(whiteImage, 1., yellowImage, 1., 0., whiteYellow);

    // Add gray filter if image is not taken during the day
    if (isDayTime == false) {
        // Gray mask
        Mat grayMask, grayImage, grayAndWhite, dst;
        std::vector<int> lowerGray = {80, 80, 80};
        std::vector<int> upperGray = {130, 130, 130};
        inRange(source, lowerGray, upperGray, grayMask);
        bitwise_and(source, source, grayImage, grayMask);

        // Blend gray, yellow and white together and return the result
        addWeighted(grayImage, 1., whiteYellow, 1., 0., dst);
        return dst;
    }

    // Return white and yellow mask if image is taken during the day
    return whiteYellow;
}

Mat LaneDetector::RegionOfInterest(Mat source) {
    /* In an ideal situation, the ROI should only contain the road lanes.
    We want to filter out all the other stuff, including things like arrow road
    markings. We try to achieve that by creating two trapezoid masks: one big
    trapezoid and a smaller one. The smaller one goes inside the bigger one. The
    pixels in the space between them will be kept and all the other pixels will
    be masked. If it goes well, the space between the two trapezoids contains
    only the lanes. */

    // Parameters big trapezoid
    float trapezoidBottomWidth = 1.0; // Width of bottom edge of trapezoid,
                                      // expressed as percentage of image width
    float trapezoidTopWidth = 0.07; // Above comment also applies here, but then
                                    // for the top edge of trapezoid
    float trapezoidHeight =
        0.5; // Height of the trapezoid expressed as percentage of image height

    // Parameters small trapezoid
    float smallBottomWidth = 0.45; // This will be added to trapezoidBottomWidth
                                   // to create a less wide bottom edge
    float smallTopWidth =
        0.3; // We multiply the percentage trapoezoidTopWidth with this
             // parameter to create a less wide top edge
    float smallHeight = 1.0; // Height of the small trapezoid expressed as
                             // percentage of height of big trapezoid

    // This parameter will make the trapezoids float just above the bottom edge
    // of the image
    float bar = 0.97;

    // Vector which holds all the points of the two trapezoids
    std::vector<Point> pts;

    // Large trapezoid
    pts.push_back(cv::Point((source.cols * (1 - trapezoidBottomWidth)) / 2,
                            source.rows * bar)); // Bottom left
    pts.push_back(
        cv::Point((source.cols * (1 - trapezoidTopWidth)) / 2,
                  source.rows - source.rows * trapezoidHeight)); // Top left
    pts.push_back(
        cv::Point(source.cols - (source.cols * (1 - trapezoidTopWidth)) / 2,
                  source.rows - source.rows * trapezoidHeight)); // Top right
    pts.push_back(
        cv::Point(source.cols - (source.cols * (1 - trapezoidBottomWidth)) / 2,
                  source.rows * bar)); // Bottom right

    // Small trapezoid
    pts.push_back(cv::Point(
        (source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2,
        source.rows * bar)); // Bottom left
    pts.push_back(cv::Point(
        (source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2,
        source.rows - source.rows * trapezoidHeight * smallHeight)); // Top left
    pts.push_back(cv::Point(
        source.cols -
            (source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2,
        source.rows -
            source.rows * trapezoidHeight * smallHeight)); // Top right
    pts.push_back(cv::Point(
        source.cols -
            (source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2,
        source.rows * bar)); // Bottom right

    std::vector<std::vector<Point>> contours;
    contours.push_back(pts);

    // Create the mask
    Mat mask = Mat::zeros(source.size(), source.type());
    fillPoly(mask, contours, Scalar(255, 255, 255));

    /* And here we basically put the mask over the source image,
    meaning we return an all black image, except for the part where the mask
    image has nonzero pixels: all the pixels in the space between the two
    trapezoids */
    Mat maskedImage;
    bitwise_and(source, mask, maskedImage);

    return maskedImage;
}

Mat LaneDetector::drawLanes(Mat source, std::vector<Vec4i> lines) {
    // Stop if there are no lines, just return original image without lines
    if (lines.size() == 0) {
        return source;
    }

    // Set drawing lanes to true
    bool drawRightLane = true;
    bool drawLeftLane = true;

    // Find lines with a slope higher than the slope threshold
    float slopeThreshold = 0.5;
    std::vector<float> slopes;
    std::vector<Vec4i> goodLines;

    for (int i = 0; i < lines.size(); i++) {
        /* Each line is represented by a 4-element vector (x_1, y_1, x_2, y_2),
        where (x_1,y_1) is the line's starting point and (x_2, y_2) is the
        ending point */
        Vec4i l = lines[i];
        double slope;

        // Calculate slope
        if (l[2] - l[0] == 0) // Avoid division by zero
        {
            slope = 999; // Basically infinte slope
        } else {
            slope = (l[3] - l[1]) / (l[2] / l[0]);
        }
        if (abs(slope) > 0.5) {
            slopes.push_back(slope);
            goodLines.push_back(l);
        }
    }

    /* Split the good lines into two categories: right and left
    The right lines have a positive slope and the left lines have a negative
    slope */
    std::vector<Vec4i> rightLines;
    std::vector<Vec4i> leftLines;
    int imgCenter = source.cols / 2;

    for (int i = 0; i < slopes.size(); i++) {
        if (slopes[i] > 0 && goodLines[i][0] > imgCenter &&
            goodLines[i][2] > imgCenter) {
            rightLines.push_back(goodLines[i]);
        }
        if (slopes[i] < 0 && goodLines[i][0] < imgCenter &&
            goodLines[i][2] < imgCenter) {
            leftLines.push_back(goodLines[i]);
        }
    }

    /* Now that we've isolated the right lane lines from the left lane lines,
    it is time to form two lane lines out of all the lines we've detected.
    A line is defined as 2 points: a starting point and an ending point.
    So up to this point the right and left lane basically consist of multiple
    hough lines. Our goal at this step is to use linear regression to find the
    two best fitting lines: one through the points at the left side to form the
    left lane and one through the points at the right side to form the right
    lane */

    // We start with the right side points
    std::vector<int> rightLinesX;
    std::vector<int> rightLinesY;
    double rightB1, rightB0; // Slope and intercept

    for (int i = 0; i < rightLines.size(); i++) {
        rightLinesX.push_back(rightLines[i][0]); // X of starting point of line
        rightLinesX.push_back(rightLines[i][2]); // X of ending point of line
        rightLinesY.push_back(rightLines[i][1]); // Y of starting point of line
        rightLinesY.push_back(rightLines[i][3]); // Y of ending point of line
    }

    if (rightLinesX.size() > 0) {
        std::vector<double> coefRight = estimateCoefficients<int, double>(
            rightLinesX, rightLinesY); // y = b1x + b0
        rightB1 = coefRight[0];
        rightB0 = coefRight[1];
    } else {
        rightB1 = 1;
        rightB0 = 1;
        drawRightLane = false;
    }

    // Now the points at the left side
    std::vector<int> leftLinesX;
    std::vector<int> leftLinesY;
    double leftB1, leftB0; // Slope and intercept

    for (int i = 0; i < leftLines.size(); i++) {
        leftLinesX.push_back(leftLines[i][0]); // X of starting point of line
        leftLinesX.push_back(leftLines[i][2]); // X of ending point of line
        leftLinesY.push_back(leftLines[i][1]); // Y of starting point of line
        leftLinesY.push_back(leftLines[i][3]); // Y of ending point of line
    }

    if (leftLinesX.size() > 0) {
        std::vector<double> coefLeft = estimateCoefficients<int, double>(
            leftLinesX, leftLinesY); // y = b1x + b0
        leftB1 = coefLeft[0];
        leftB0 = coefLeft[1];
    } else {
        leftB1 = 1;
        leftB0 = 1;
        drawLeftLane = false;
    }

    /* Now we need to find the two points for the right and left lane:
    starting points and ending points */

    int y1 = source.rows; // Y coordinate of starting point of both the left and
                          // right lane

    /* 0.5 = trapezoidHeight (see RegionOfInterest), we set the y coordinate of
    the ending point below the trapezoid height (0.4) to draw shorter lanes. I
    think that looks nicer. */

    int y2 =
        source.rows *
        (1 -
         0.4); // Y coordinate of ending point of both the left and right lane

    // y = b1x + b0 --> x = (y - b0) / b1
    int rightX1 = (y1 - rightB0) /
                  rightB1; // X coordinate of starting point of right lane
    int rightX2 =
        (y2 - rightB0) / rightB1; // X coordinate of ending point of right lane

    int leftX1 =
        (y1 - leftB0) / leftB1; // X coordinate of starting point of left lane
    int leftX2 =
        (y2 - leftB0) / leftB1; // X coordinate of ending point of left lane

    /* If the ending point of the right lane is on the left side of the left
    lane (or vice versa), return source image without drawings, because this
    should not be happening in real life. */
    if (rightX2 < leftX2 || leftX2 > rightX2) {
        return source;
    }

    // Create the mask
    Mat mask = Mat::zeros(source.size(), source.type());

    // Draw lines and fill poly made up of the four points described above if
    // both bools are true
    Mat dst; // Holds blended image
    if (drawRightLane == true && drawLeftLane == true) {
        line(source, Point(rightX1, y1), Point(rightX2, y2), Scalar(255, 0, 0),
             7);
        line(source, Point(leftX1, y1), Point(leftX2, y2), Scalar(255, 0, 0),
             7);

        Point pts[4] = {
            Point(leftX1, y1),  // Starting point left lane
            Point(leftX2, y2),  // Ending point left lane
            Point(rightX2, y2), // Ending point right lane
            Point(rightX1, y1)  // Starting point right lane
        };

        fillConvexPoly(mask, pts, 4,
                       Scalar(235, 229, 52)); // Color is light blue

        // Blend the mask and source image together
        addWeighted(source, 0.9, mask, 0.3, 0.0, dst);

        // Return blended image
        return dst;
    }

    return source; // Return source if drawing lanes did not happen
}

std::vector<Vec4i> LaneDetector::houghLines(Mat canny, Mat source,
                                            bool drawHough) {
    double rho = 2; // Distance resolution in pixels of the Hough grid
    double theta =
        1 * M_PI / 180; // Angular resolution in radians of the Hough grid
    int thresh =
        15; // Minimum number of votes (intersections in Hough grid cell)
    double minLineLength = 10; // Minimum number of pixels making up a line
    double maxGapLength =
        20; // Maximum gap in pixels between connectable line segments

    std::vector<Vec4i> linesP; // Will hold the results of the detection
    HoughLinesP(canny, linesP, rho, theta, thresh, minLineLength, maxGapLength);

    if (drawHough == true) {
        for (size_t i = 0; i < linesP.size(); i++) {
            Vec4i l = linesP[i];
            line(source, Point(l[0], l[1]), Point(l[2], l[3]),
                 Scalar(0, 0, 255), 3, LINE_AA);
        }
        imshow("Hough Lines", source);
        waitKey(0);
    }

    return linesP;
}

bool LaneDetector::isDayTime(Mat source) {
    /* I've noticed that, in general, daytime images/videos require different
    color filters than nighttime images/videos. For example, in darker light it
    is better to add a gray color filter in addition to the white and yellow one
  */

    Scalar s = mean(source); // Mean pixel values

    /* I chose these cut off values by looking at the mean pixel values of
    multiple daytime and nighttime images */
    if (s[0] < 30 || s[1] < 33 && s[2] < 30) {
        return false;
    }

    return true;
}