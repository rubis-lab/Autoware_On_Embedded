
/// \file TrajectoryDynamicCosts.cpp
/// \brief Calculate collision costs for roll out trajectory for free trajectory evaluation for OpenPlanner local planner version 1.5+
/// \author Hatem Darweesh
/// \date Jan 14, 2018

#include "op_planner/TrajectoryDynamicCosts.h"
#include "op_planner/MatrixOperations.h"
#include "float.h"

// #define DEBUG_ENABLE

namespace PlannerHNS
{


TrajectoryDynamicCosts::TrajectoryDynamicCosts()
{
  m_PrevCostIndex = -1;
  m_PrevSelectedIndex = -1;
  m_WeightPriority = 0.9;
  m_WeightTransition = 0.9;
  m_WeightLong = 1.0;
  m_WeightLat = 1.2;
  m_WeightLaneChange = 0.0;
  m_LateralSkipDistance = 5;

  m_CollisionTimeDiff = 6.0; //seconds
  m_PrevIndex = -1;
}

TrajectoryDynamicCosts::~TrajectoryDynamicCosts()
{
}

TrajectoryCost TrajectoryDynamicCosts::DoOneStepDynamic(const vector<vector<WayPoint> >& rollOuts,
    const vector<WayPoint>& totalPaths, const WayPoint& currState,
    const PlanningParams& params, const CAR_BASIC_INFO& carInfo, const VehicleState& vehicleState,
    const std::vector<PlannerHNS::DetectedObject>& obj_list, const int& iCurrentIndex)
{
  TrajectoryCost bestTrajectory;
  bestTrajectory.bBlocked = true;
  bestTrajectory.closest_obj_distance = params.horizonDistance;
  bestTrajectory.closest_obj_velocity = 0;
  bestTrajectory.index = -1;

  double critical_lateral_distance   =  carInfo.width/2.0 + params.horizontalSafetyDistancel;
  double critical_long_front_distance =  carInfo.wheel_base/2.0 + carInfo.length/2.0 + params.verticalSafetyDistance;
  double critical_long_back_distance   =  carInfo.length/2.0 + params.verticalSafetyDistance - carInfo.wheel_base/2.0;

  int currIndex = -1;
  if(iCurrentIndex >=0 && iCurrentIndex < rollOuts.size())
    currIndex  = iCurrentIndex;
  else
    currIndex = GetCurrentRollOutIndex(totalPaths, currState, params);

  InitializeCosts(rollOuts, params);

  InitializeSafetyPolygon(currState, carInfo, vehicleState, critical_lateral_distance, critical_long_front_distance, critical_long_back_distance);

  CalculateTransitionCosts(m_TrajectoryCosts, currIndex, params);

  CalculateLateralAndLongitudinalCostsDynamic(obj_list, rollOuts, totalPaths,  currState, params, carInfo, vehicleState, critical_lateral_distance, critical_long_front_distance, critical_long_back_distance);

  NormalizeCosts(m_TrajectoryCosts);

  int smallestIndex = -1;
  double smallestCost = DBL_MAX;
  double smallestDistance = DBL_MAX;
  double velo_of_next = 0;
  bool bAllFree = true;

  //cout << "Trajectory Costs Log : CurrIndex: " << currIndex << " --------------------- " << endl;
  for(unsigned int ic = 0; ic < m_TrajectoryCosts.size(); ic++)
  {
    if(!m_TrajectoryCosts.at(ic).bBlocked && m_TrajectoryCosts.at(ic).cost < smallestCost)
    {
      smallestCost = m_TrajectoryCosts.at(ic).cost;
      smallestIndex = ic;
    }

    if(m_TrajectoryCosts.at(ic).closest_obj_distance < smallestDistance)
    {
      smallestDistance = m_TrajectoryCosts.at(ic).closest_obj_distance;
      velo_of_next = m_TrajectoryCosts.at(ic).closest_obj_velocity;
    }

    if(m_TrajectoryCosts.at(ic).bBlocked)
      bAllFree = false;
  }

  if(bAllFree && smallestIndex >=0)
    smallestIndex = params.rollOutNumber/2;


  if(smallestIndex == -1)
  {
    bestTrajectory.bBlocked = true;
    bestTrajectory.lane_index = 0;
    bestTrajectory.index = m_PrevCostIndex;
    bestTrajectory.closest_obj_distance = smallestDistance;
    bestTrajectory.closest_obj_velocity = velo_of_next;
  }
  else if(smallestIndex >= 0)
  {
    bestTrajectory = m_TrajectoryCosts.at(smallestIndex);
  }

  m_PrevIndex = currIndex;

  //std::cout << "Current Selected Index : " << bestTrajectory.index << std::endl;
  return bestTrajectory;
}

TrajectoryCost TrajectoryDynamicCosts::DoOneStepStatic(const vector<vector<WayPoint> >& rollOuts,
    const vector<WayPoint>& totalPaths, const WayPoint& currState,
    const PlanningParams& params, const CAR_BASIC_INFO& carInfo, const VehicleState& vehicleState,
    const std::vector<PlannerHNS::DetectedObject>& obj_list, const PlannerHNS::STATE_TYPE& state, const int& iCurrentIndex)
{
  TrajectoryCost bestTrajectory;
  bestTrajectory.bBlocked = true;
  bestTrajectory.closest_obj_distance = params.horizonDistance;
  bestTrajectory.closest_obj_velocity = 0;
  bestTrajectory.index = -1;

  // get parameter
  m_WeightPriority = params.weightPriority;
  m_WeightTransition = params.weightTransition;
  m_WeightLong = params.weightLong;
  m_WeightLat = params.weightLat;
  m_LateralSkipDistance = params.LateralSkipDistance;

  RelativeInfo obj_info;
  PlanningHelpers::GetRelativeInfo(totalPaths, currState, obj_info);
  int currIndex = params.rollOutNumber/2 + floor(obj_info.perp_distance/params.rollOutDensity);
  //std::cout <<  "Current Index: " << currIndex << std::endl;
  if(currIndex < 0)
    currIndex = 0;
  else if(currIndex > params.rollOutNumber)
    currIndex = params.rollOutNumber;

  // Calculate lane change cost: Scoring the cost by the distance between current path and candidate path
  m_TrajectoryCosts.clear();
  if(rollOuts.size()>0)
  {
    TrajectoryCost tc;
    int centralIndex = params.rollOutNumber/2;
    tc.lane_index = 0;
    for(unsigned int it=0; it< rollOuts.size(); it++)
    {
      tc.index = it;
      tc.relative_index = it - centralIndex; // index based on center (central index is 0)
      tc.distance_from_center = params.rollOutDensity*tc.relative_index;
      tc.priority_cost = fabs(tc.distance_from_center); // priority_cost = distance from center
      tc.closest_obj_distance = params.horizonDistance;
      if(rollOuts.at(it).size() > 0)
          tc.lane_change_cost = rollOuts.at(it).at(0).laneChangeCost;
      m_TrajectoryCosts.push_back(tc);
    }
  }

  // std::cout << m_PrevSelectedIndex << std::endl;

  if(m_PrevSelectedIndex == -1){
    CalculateTransitionCosts(m_TrajectoryCosts, currIndex, params);
  }
  else{
    CalculateTransitionCosts(m_TrajectoryCosts, m_PrevSelectedIndex, params);
  }
  //end

  // Calculate trajectory cost with object
  WayPoint p;
  m_AllContourPoints.clear();
  for(unsigned int io=0; io<obj_list.size(); io++) // Object in object list
  {
    // for(unsigned int icon=0; icon < obj_list.at(io).contour.size(); icon++) // Point in contour
    // {
    //   p.pos = obj_list.at(io).contour.at(icon);
    //   p.v = obj_list.at(io).center.v;
    //   p.id = io;
    //   p.cost = sqrt(obj_list.at(io).w*obj_list.at(io).w + obj_list.at(io).l*obj_list.at(io).l);
    //   m_AllContourPoints.push_back(p);
    // }

    // Only add centroid
    p.pos = obj_list.at(io).center.pos;
    p.v = obj_list.at(io).center.v;
    p.id = io;
    m_AllContourPoints.push_back(p);
  }

  CalculateLateralAndLongitudinalCostsStatic(m_TrajectoryCosts, rollOuts, totalPaths, currState, m_AllContourPoints, params, carInfo, vehicleState);

  NormalizeCosts(m_TrajectoryCosts);

  int smallestIndex = -1;
  double smallestCost = DBL_MAX;
  double smallestDistance = DBL_MAX;
  double velo_of_next = 0;

  bool bAllFree = true;

  for(unsigned int ic = 0; ic < m_TrajectoryCosts.size(); ic++)
  {
    if(!m_TrajectoryCosts.at(ic).bBlocked && m_TrajectoryCosts.at(ic).cost < smallestCost)
    {
      smallestCost = m_TrajectoryCosts.at(ic).cost;
      smallestIndex = ic;
    }

    if(m_TrajectoryCosts.at(ic).closest_obj_distance < smallestDistance)
    {
      smallestDistance = m_TrajectoryCosts.at(ic).closest_obj_distance;
      velo_of_next = m_TrajectoryCosts.at(ic).closest_obj_velocity;
    }

    if(m_TrajectoryCosts.at(ic).lateral_cost > 0){
      bAllFree = false;
    }
  }

  #ifdef DEBUG_ENABLE
    std::cout << "Index : " << smallestIndex << std::endl;
  #endif

  // Enable when needed
  // if(bAllFree && smallestIndex >= 0){
  //   smallestIndex = params.rollOutNumber/2;
  // }

  ////// Change Lane when turn left / right
  // Calculate Turn Angle
  double turn_angle = 0;

  if(m_PrevSelectedIndex != -1)
    turn_angle = CalculateTurnAngle(rollOuts.at(m_PrevSelectedIndex), currState, 50);

  // std::cout << "b_all_free : " << bAllFree << " , t_a : " << turn_angle << std::endl;

  // Keep state when Intersection state
  if(state == PlannerHNS::INTERSECTION_STATE){
    smallestIndex = m_PrevSelectedIndex;
  }
  // For Left Turn
  else if(bAllFree && turn_angle > 45){
    smallestIndex = 0;
  }
  // For Right Turn
  else if(bAllFree && turn_angle < -45 && params.rollOutNumber > 0){
    smallestIndex = params.rollOutNumber - 1;
  }

  if(smallestIndex == -1)
  {
    bestTrajectory.bBlocked = true;
    bestTrajectory.lane_index = m_PrevSelectedIndex;
    bestTrajectory.index = m_PrevSelectedIndex;
    bestTrajectory.closest_obj_distance = smallestDistance;
    bestTrajectory.closest_obj_velocity = velo_of_next;
  }
  else if(smallestIndex >= 0)
  {
    bestTrajectory = m_TrajectoryCosts.at(smallestIndex);
    m_PrevSelectedIndex = smallestIndex;
  }

  m_PrevIndex = currIndex;
  return bestTrajectory;
}

TrajectoryCost TrajectoryDynamicCosts::DoOneStep(const vector<vector<vector<WayPoint> > >& rollOuts,
    const vector<vector<WayPoint> >& totalPaths, const WayPoint& currState, const int& currIndex,
    const int& currLaneIndex,
    const PlanningParams& params, const CAR_BASIC_INFO& carInfo, const VehicleState& vehicleState,
    const std::vector<PlannerHNS::DetectedObject>& obj_list)
{
  TrajectoryCost bestTrajectory;
  bestTrajectory.bBlocked = true;
  bestTrajectory.closest_obj_distance = params.horizonDistance;
  bestTrajectory.closest_obj_velocity = 0;
  bestTrajectory.index = -1;

  if(!ValidateRollOutsInput(rollOuts) || rollOuts.size() != totalPaths.size()) return bestTrajectory;

  if(m_PrevCostIndex == -1)
    m_PrevCostIndex = params.rollOutNumber/2;

  m_TrajectoryCosts.clear();

  for(unsigned int il = 0; il < rollOuts.size(); il++)
  {
    if(rollOuts.at(il).size()>0 && rollOuts.at(il).at(0).size()>0)
    {
      vector<TrajectoryCost> costs = CalculatePriorityAndLaneChangeCosts(rollOuts.at(il), il, params);
      m_TrajectoryCosts.insert(m_TrajectoryCosts.end(), costs.begin(), costs.end());
    }
  }

  CalculateTransitionCosts(m_TrajectoryCosts, currIndex, params);


  WayPoint p;
  m_AllContourPoints.clear();
  for(unsigned int io=0; io<obj_list.size(); io++)
  {
    for(unsigned int icon=0; icon < obj_list.at(io).contour.size(); icon++)
    {
      p.pos = obj_list.at(io).contour.at(icon);
      p.v = obj_list.at(io).center.v;
      p.id = io;
      p.cost = sqrt(obj_list.at(io).w*obj_list.at(io).w + obj_list.at(io).l*obj_list.at(io).l);
      m_AllContourPoints.push_back(p);
    }
  }

  CalculateLateralAndLongitudinalCosts(m_TrajectoryCosts, rollOuts, totalPaths, currState, m_AllContourPoints, params, carInfo, vehicleState);

  NormalizeCosts(m_TrajectoryCosts);

  int smallestIndex = -1;
  double smallestCost = DBL_MAX;
  double smallestDistance = DBL_MAX;
  double velo_of_next = 0;

  //cout << "Trajectory Costs Log : CurrIndex: " << currIndex << " --------------------- " << endl;
  for(unsigned int ic = 0; ic < m_TrajectoryCosts.size(); ic++)
  {
    //cout << m_TrajectoryCosts.at(ic).ToString();
    if(!m_TrajectoryCosts.at(ic).bBlocked && m_TrajectoryCosts.at(ic).cost < smallestCost)
    {
      smallestCost = m_TrajectoryCosts.at(ic).cost;
      smallestIndex = ic;
    }

    if(m_TrajectoryCosts.at(ic).closest_obj_distance < smallestDistance)
    {
      smallestDistance = m_TrajectoryCosts.at(ic).closest_obj_distance;
      velo_of_next = m_TrajectoryCosts.at(ic).closest_obj_velocity;
    }
  }

  //cout << "Smallest Distance: " <<  smallestDistance << "------------------------------------------------------------- " << endl;

  //All is blocked !
  if(smallestIndex == -1 && m_PrevCostIndex < (int)m_TrajectoryCosts.size())
  {
    bestTrajectory.bBlocked = true;
    bestTrajectory.lane_index = currLaneIndex;
    bestTrajectory.index = currIndex;
    bestTrajectory.closest_obj_distance = smallestDistance;
    bestTrajectory.closest_obj_velocity = velo_of_next;
    //bestTrajectory.index = smallestIndex;
  }
  else if(smallestIndex >= 0)
  {
    bestTrajectory = m_TrajectoryCosts.at(smallestIndex);
    //bestTrajectory.index = smallestIndex;
  }

//  cout << "smallestI: " <<  smallestIndex << ", C_Size: " << m_TrajectoryCosts.size()
//      << ", LaneI: " << bestTrajectory.lane_index << "TrajI: " << bestTrajectory.index
//      << ", prevSmalI: " << m_PrevCostIndex << ", distance: " << bestTrajectory.closest_obj_distance
//      << ", Blocked: " << bestTrajectory.bBlocked
//      << endl;

  m_PrevCostIndex = smallestIndex;

  return bestTrajectory;
}

void TrajectoryDynamicCosts::CalculateLateralAndLongitudinalCostsStatic(vector<TrajectoryCost>& trajectoryCosts,
    const vector<vector<WayPoint> >& rollOuts, const vector<WayPoint>& totalPaths,
    const WayPoint& currState, const vector<WayPoint>& contourPoints, const PlanningParams& params,
    const CAR_BASIC_INFO& carInfo, const VehicleState& vehicleState)
{
  double critical_lateral_distance =  carInfo.width/2.0 + params.horizontalSafetyDistancel;
  double critical_long_front_distance =  carInfo.wheel_base/2.0 + carInfo.length/2.0 + params.verticalSafetyDistance;
  double critical_long_back_distance =  carInfo.length/2.0 + params.verticalSafetyDistance - carInfo.wheel_base/2.0;

  PlannerHNS::Mat3 invRotationMat(currState.pos.a-M_PI_2);
  PlannerHNS::Mat3 invTranslationMat(currState.pos.x, currState.pos.y);

  double corner_slide_distance = critical_lateral_distance/2.0;
  double ratio_to_angle = corner_slide_distance/carInfo.max_steer_angle;
  double slide_distance = vehicleState.steer * ratio_to_angle;

  GPSPoint bottom_left(-critical_lateral_distance ,-critical_long_back_distance,  currState.pos.z, 0);
  GPSPoint bottom_right(critical_lateral_distance, -critical_long_back_distance,  currState.pos.z, 0);

  GPSPoint top_right_car(critical_lateral_distance, carInfo.wheel_base/3.0 + carInfo.length/3.0,  currState.pos.z, 0);
  GPSPoint top_left_car(-critical_lateral_distance, carInfo.wheel_base/3.0 + carInfo.length/3.0, currState.pos.z, 0);

  GPSPoint top_right(critical_lateral_distance - slide_distance, critical_long_front_distance,  currState.pos.z, 0);
  GPSPoint top_left(-critical_lateral_distance - slide_distance , critical_long_front_distance, currState.pos.z, 0);

  bottom_left = invRotationMat*bottom_left;
  bottom_left = invTranslationMat*bottom_left;

  top_right = invRotationMat*top_right;
  top_right = invTranslationMat*top_right;

  bottom_right = invRotationMat*bottom_right;
  bottom_right = invTranslationMat*bottom_right;

  top_left = invRotationMat*top_left;
  top_left = invTranslationMat*top_left;

  top_right_car = invRotationMat*top_right_car;
  top_right_car = invTranslationMat*top_right_car;

  top_left_car = invRotationMat*top_left_car;
  top_left_car = invTranslationMat*top_left_car;

  m_SafetyBorder.points.clear();
  m_SafetyBorder.points.push_back(bottom_left) ;
  m_SafetyBorder.points.push_back(bottom_right) ;
  m_SafetyBorder.points.push_back(top_right_car) ;
  m_SafetyBorder.points.push_back(top_right) ;
  m_SafetyBorder.points.push_back(top_left) ;
  m_SafetyBorder.points.push_back(top_left_car);

  #ifdef DEBUG_ENABLE
    std::cout << "points num : " << contourPoints.size() << std::endl;
  #endif

  int iCostIndex = 0;
  if(rollOuts.size() > 0 && rollOuts.at(0).size()>0)
  {
    RelativeInfo car_info;
    PlanningHelpers::GetRelativeInfo(totalPaths, currState, car_info);

    for(unsigned int it=0; it< rollOuts.size(); it++)
    {
      #ifdef DEBUG_ENABLE
      int unskipped = 0;
      #endif

      int skip_id = -1;
      for(unsigned int icon = 0; icon < contourPoints.size(); icon++)
      {
        // if(skip_id == contourPoints.at(icon).id)
        //   continue;

        RelativeInfo obj_info;
        PlanningHelpers::GetRelativeInfo(totalPaths, contourPoints.at(icon), obj_info);
        double longitudinalDist = PlanningHelpers::GetExactDistanceOnTrajectory(totalPaths, car_info, obj_info); // Caculate cumulated distance between car and obj in path
        if(obj_info.iFront == 0 && longitudinalDist > 0)
          longitudinalDist = -longitudinalDist;

        // double direct_distance = hypot(obj_info.perp_point.pos.y-contourPoints.at(icon).pos.y, obj_info.perp_point.pos.x-contourPoints.at(icon).pos.x); // Caculate direct distance

        // Original
        // if(contourPoints.at(icon).v < params.minSpeed && direct_distance > (m_LateralSkipDistance+contourPoints.at(icon).cost))
        // {
        //   skip_id = contourPoints.at(icon).id;
        //   continue;
        // }

        // std::cout << obj_info.perp_distance << std::endl;
        // std::cout << rollOuts.size() << " " << rollOuts.size() /2 << std::endl;
        // std::cout << ((rollOuts.size() / 2) * params.rollOutDensity) * (-1) << std::endl;

        double distance_from_center = trajectoryCosts.at(iCostIndex).distance_from_center; // Disetance between center and trajectory

        double lateralDist = fabs(obj_info.perp_distance - distance_from_center);

        // std::cout << "long : " << longitudinalDist << ", lat : " << lateralDist << std::endl;

        if(longitudinalDist < 0 ||
          longitudinalDist > 30 ||
          obj_info.perp_distance < (((rollOuts.size() - 1) / 2) * params.rollOutDensity + critical_lateral_distance / 2) * (-1) ||
          obj_info.perp_distance > ((rollOuts.size() - 1) / 2 + 1) * params.rollOutDensity + critical_lateral_distance / 2)
          // obj_info.perp_distance < (((rollOuts.size() - 1) / 2) * params.rollOutDensity) * (-1) ||
          // obj_info.perp_distance > ((rollOuts.size() - 1) / 2 + 1) * params.rollOutDensity)
        {
          continue;
        }

        #ifdef DEBUG_ENABLE
        unskipped++;
        #endif

        // Original
        // if(longitudinalDist < -carInfo.length || longitudinalDist > params.minFollowingDistance || lateralDist > m_LateralSkipDistance)
        // {
        //   continue;
        // }

        if(m_SafetyBorder.PointInsidePolygon(m_SafetyBorder, contourPoints.at(icon).pos) == true)
          trajectoryCosts.at(iCostIndex).bBlocked = true;

        // Disabled bj hjw
        if(lateralDist <= 1.5
            && longitudinalDist >= -5
            && longitudinalDist < 30)
          trajectoryCosts.at(it).bBlocked = true;

        // Original
        // if(lateralDist <= critical_lateral_distance
        //     && longitudinalDist >= -carInfo.length/1.5
        //     && longitudinalDist < params.minFollowingDistance)
        //   trajectoryCosts.at(iCostIndex).bBlocked = true;

        if(lateralDist != 0)
          trajectoryCosts.at(iCostIndex).lateral_cost += 1.0/lateralDist;

        if(longitudinalDist != 0)
          trajectoryCosts.at(iCostIndex).longitudinal_cost += 1.0/fabs(longitudinalDist);

        if(longitudinalDist > 0 && longitudinalDist < trajectoryCosts.at(iCostIndex).closest_obj_distance)
        {
          trajectoryCosts.at(iCostIndex).closest_obj_distance = longitudinalDist;
          trajectoryCosts.at(iCostIndex).closest_obj_velocity = contourPoints.at(icon).v;
        }
      }

      #ifdef DEBUG_ENABLE
      // std::cout << trajectoryCosts.at(iCostIndex).longitudinal_cost << " " << trajectoryCosts.at(iCostIndex).lateral_cost << ", ";
      std::cout << unskipped << " ";
      #endif

      // Calculate lateral/logitudinal cost, disdtance and velocity
      iCostIndex++;
    }
    #ifdef DEBUG_ENABLE
    std::cout << std::endl;
    #endif
  }
}

void TrajectoryDynamicCosts::CalculateLateralAndLongitudinalCosts(vector<TrajectoryCost>& trajectoryCosts,
    const vector<vector<vector<WayPoint> > >& rollOuts, const vector<vector<WayPoint> >& totalPaths,
    const WayPoint& currState, const vector<WayPoint>& contourPoints, const PlanningParams& params,
    const CAR_BASIC_INFO& carInfo, const VehicleState& vehicleState)
{
  double critical_lateral_distance =  carInfo.width/2.0 + params.horizontalSafetyDistancel;
  double critical_long_front_distance =  carInfo.wheel_base/2.0 + carInfo.length/2.0 + params.verticalSafetyDistance;
  double critical_long_back_distance =  carInfo.length/2.0 + params.verticalSafetyDistance - carInfo.wheel_base/2.0;
  int iCostIndex = 0;

  PlannerHNS::Mat3 invRotationMat(currState.pos.a-M_PI_2);
  PlannerHNS::Mat3 invTranslationMat(currState.pos.x, currState.pos.y);

  double corner_slide_distance = critical_lateral_distance/2.0;
  double ratio_to_angle = corner_slide_distance/carInfo.max_steer_angle;
  double slide_distance = vehicleState.steer * ratio_to_angle;

  GPSPoint bottom_left(-critical_lateral_distance ,-critical_long_back_distance,  currState.pos.z, 0);
  GPSPoint bottom_right(critical_lateral_distance, -critical_long_back_distance,  currState.pos.z, 0);

  GPSPoint top_right_car(critical_lateral_distance, carInfo.wheel_base/3.0 + carInfo.length/3.0,  currState.pos.z, 0);
  GPSPoint top_left_car(-critical_lateral_distance, carInfo.wheel_base/3.0 + carInfo.length/3.0, currState.pos.z, 0);

  GPSPoint top_right(critical_lateral_distance - slide_distance, critical_long_front_distance,  currState.pos.z, 0);
  GPSPoint top_left(-critical_lateral_distance - slide_distance , critical_long_front_distance, currState.pos.z, 0);

  bottom_left = invRotationMat*bottom_left;
  bottom_left = invTranslationMat*bottom_left;

  top_right = invRotationMat*top_right;
  top_right = invTranslationMat*top_right;

  bottom_right = invRotationMat*bottom_right;
  bottom_right = invTranslationMat*bottom_right;

  top_left = invRotationMat*top_left;
  top_left = invTranslationMat*top_left;

  top_right_car = invRotationMat*top_right_car;
  top_right_car = invTranslationMat*top_right_car;

  top_left_car = invRotationMat*top_left_car;
  top_left_car = invTranslationMat*top_left_car;

  m_SafetyBorder.points.clear();
  m_SafetyBorder.points.push_back(bottom_left) ;
  m_SafetyBorder.points.push_back(bottom_right) ;
  m_SafetyBorder.points.push_back(top_right_car) ;
  m_SafetyBorder.points.push_back(top_right) ;
  m_SafetyBorder.points.push_back(top_left) ;
  m_SafetyBorder.points.push_back(top_left_car) ;

  for(unsigned int il=0; il < rollOuts.size(); il++)
  {
    if(rollOuts.at(il).size() > 0 && rollOuts.at(il).at(0).size()>0)
    {
      RelativeInfo car_info;
      PlanningHelpers::GetRelativeInfo(totalPaths.at(il), currState, car_info);


      for(unsigned int it=0; it< rollOuts.at(il).size(); it++)
      {
        int skip_id = -1;
        for(unsigned int icon = 0; icon < contourPoints.size(); icon++)
        {
          if(skip_id == contourPoints.at(icon).id)
            continue;

          RelativeInfo obj_info;
          PlanningHelpers::GetRelativeInfo(totalPaths.at(il), contourPoints.at(icon), obj_info);
          double longitudinalDist = PlanningHelpers::GetExactDistanceOnTrajectory(totalPaths.at(il), car_info, obj_info);
          if(obj_info.iFront == 0 && longitudinalDist > 0)
            longitudinalDist = -longitudinalDist;

          double direct_distance = hypot(obj_info.perp_point.pos.y-contourPoints.at(icon).pos.y, obj_info.perp_point.pos.x-contourPoints.at(icon).pos.x);
          if(contourPoints.at(icon).v < params.minSpeed && direct_distance > (m_LateralSkipDistance+contourPoints.at(icon).cost))
          {
            skip_id = contourPoints.at(icon).id;
            continue;
          }

          double close_in_percentage = 1;
//          close_in_percentage = ((longitudinalDist- critical_long_front_distance)/params.rollInMargin)*4.0;
//
//          if(close_in_percentage <= 0 || close_in_percentage > 1) close_in_percentage = 1;

          double distance_from_center = trajectoryCosts.at(iCostIndex).distance_from_center;

          if(close_in_percentage < 1)
            distance_from_center = distance_from_center - distance_from_center * (1.0-close_in_percentage);

          double lateralDist = fabs(obj_info.perp_distance - distance_from_center);

          if(longitudinalDist < -carInfo.length || longitudinalDist > params.minFollowingDistance || lateralDist > m_LateralSkipDistance)
          {
            continue;
          }

          longitudinalDist = longitudinalDist - critical_long_front_distance;

          if(m_SafetyBorder.PointInsidePolygon(m_SafetyBorder, contourPoints.at(icon).pos) == true)
            trajectoryCosts.at(iCostIndex).bBlocked = true;

          if(lateralDist <= critical_lateral_distance
              && longitudinalDist >= -carInfo.length/1.5
              && longitudinalDist < params.minFollowingDistance)
            trajectoryCosts.at(iCostIndex).bBlocked = true;


          if(lateralDist != 0)
            trajectoryCosts.at(iCostIndex).lateral_cost += 1.0/lateralDist;

          if(longitudinalDist != 0)
            trajectoryCosts.at(iCostIndex).longitudinal_cost += 1.0/fabs(longitudinalDist);


          if(longitudinalDist >= -critical_long_front_distance && longitudinalDist < trajectoryCosts.at(iCostIndex).closest_obj_distance)
          {
            trajectoryCosts.at(iCostIndex).closest_obj_distance = longitudinalDist;
            trajectoryCosts.at(iCostIndex).closest_obj_velocity = contourPoints.at(icon).v;
          }
        }

        iCostIndex++;
      }
    }
  }
}

void TrajectoryDynamicCosts::NormalizeCosts(vector<TrajectoryCost>& trajectoryCosts)
{
  //Normalize costs
  double totalPriorities = 0;
  double totalChange = 0;
  double totalLateralCosts = 0;
  double totalLongitudinalCosts = 0;
  double transitionCosts = 0;

  for(unsigned int ic = 0; ic< trajectoryCosts.size(); ic++)
  {
    totalPriorities += trajectoryCosts.at(ic).priority_cost;
    transitionCosts += trajectoryCosts.at(ic).transition_cost;
  }

  for(unsigned int ic = 0; ic< trajectoryCosts.size(); ic++)
  {
    totalChange += trajectoryCosts.at(ic).lane_change_cost;
    totalLateralCosts += trajectoryCosts.at(ic).lateral_cost;
    totalLongitudinalCosts += trajectoryCosts.at(ic).longitudinal_cost;
  }

//  cout << "------ Normalizing Step " << endl;
  for(unsigned int ic = 0; ic< trajectoryCosts.size(); ic++)
  {
    if(totalPriorities != 0)
      trajectoryCosts.at(ic).priority_cost = trajectoryCosts.at(ic).priority_cost / totalPriorities;
    else
      trajectoryCosts.at(ic).priority_cost = 0;

    if(transitionCosts != 0)
      trajectoryCosts.at(ic).transition_cost = trajectoryCosts.at(ic).transition_cost / transitionCosts;
    else
      trajectoryCosts.at(ic).transition_cost = 0;

    if(totalChange != 0)
      trajectoryCosts.at(ic).lane_change_cost = trajectoryCosts.at(ic).lane_change_cost / totalChange;
    else
      trajectoryCosts.at(ic).lane_change_cost = 0;

    if(totalLateralCosts != 0)
      trajectoryCosts.at(ic).lateral_cost = trajectoryCosts.at(ic).lateral_cost / totalLateralCosts;
    else
      trajectoryCosts.at(ic).lateral_cost = 0;

    if(totalLongitudinalCosts != 0)
      trajectoryCosts.at(ic).longitudinal_cost = trajectoryCosts.at(ic).longitudinal_cost / totalLongitudinalCosts;
    else
      trajectoryCosts.at(ic).longitudinal_cost = 0;

    trajectoryCosts.at(ic).cost = (m_WeightPriority*trajectoryCosts.at(ic).priority_cost + m_WeightTransition*trajectoryCosts.at(ic).transition_cost + m_WeightLat*trajectoryCosts.at(ic).lateral_cost + m_WeightLong*trajectoryCosts.at(ic).longitudinal_cost)/4.0;

  #ifdef DEBUG_ENABLE
   std::cout << "Index: " << ic
           << ", Priority: " << trajectoryCosts.at(ic).priority_cost
           << ", Transition: " << trajectoryCosts.at(ic).transition_cost
           << ", Lat: " << trajectoryCosts.at(ic).lateral_cost
           << ", Long: " << trajectoryCosts.at(ic).longitudinal_cost
           << ", Change: " << trajectoryCosts.at(ic).lane_change_cost
           << ", Avg: " << trajectoryCosts.at(ic).cost
           << ", Blocked : " << trajectoryCosts.at(ic).bBlocked
           << std::endl;
  #endif
  }

  #ifdef DEBUG_ENABLE
  std::cout << "------------------------ " << std::endl;
  #endif
}

vector<TrajectoryCost> TrajectoryDynamicCosts::CalculatePriorityAndLaneChangeCosts(const vector<vector<WayPoint> >& laneRollOuts,
    const int& lane_index, const PlanningParams& params)
{
  vector<TrajectoryCost> costs;
  TrajectoryCost tc;
  int centralIndex = params.rollOutNumber/2;

  tc.lane_index = lane_index;
  for(unsigned int it=0; it< laneRollOuts.size(); it++)
  {
    tc.index = it;
    tc.relative_index = it - centralIndex;
    tc.distance_from_center = params.rollOutDensity*tc.relative_index;
    tc.priority_cost = fabs(tc.distance_from_center);
    tc.closest_obj_distance = params.horizonDistance;
    tc.lane_change_cost = laneRollOuts.at(it).at(0).laneChangeCost;

//    if(laneRollOuts.at(it).at(0).bDir == FORWARD_LEFT_DIR || laneRollOuts.at(it).at(0).bDir == FORWARD_RIGHT_DIR)
//      tc.lane_change_cost = 1;
//    else if(laneRollOuts.at(it).at(0).bDir == BACKWARD_DIR || laneRollOuts.at(it).at(0).bDir == BACKWARD_RIGHT_DIR || laneRollOuts.at(it).at(0).bDir == BACKWARD_LEFT_DIR)
//      tc.lane_change_cost = 2;
//    else
//      tc.lane_change_cost = 0;

    costs.push_back(tc);
  }

  return costs;
}

void TrajectoryDynamicCosts::CalculateTransitionCosts(vector<TrajectoryCost>& trajectoryCosts, const int& currTrajectoryIndex, const PlanningParams& params)
{
  for(int ic = 0; ic< trajectoryCosts.size(); ic++)
  {
    trajectoryCosts.at(ic).transition_cost = fabs(params.rollOutDensity * (ic - currTrajectoryIndex));
  }
}

/**
 * @brief Validate input, each trajectory must have at least 1 way point
 * @param rollOuts
 * @return true if data isvalid for cost calculation
 */
bool TrajectoryDynamicCosts::ValidateRollOutsInput(const vector<vector<vector<WayPoint> > >& rollOuts)
{
  if(rollOuts.size()==0)
    return false;

  for(unsigned int il = 0; il < rollOuts.size(); il++)
  {
    if(rollOuts.at(il).size() == 0)
      return false;
  }

  return true;
}

void TrajectoryDynamicCosts::CalculateIntersectionVelocities(const std::vector<PlannerHNS::WayPoint>& path, const PlannerHNS::DetectedObject& obj, const WayPoint& currPose, const CAR_BASIC_INFO& carInfo, const double& c_lateral_d, WayPoint& collisionPoint, TrajectoryCost& trajectoryCosts)
{
  trajectoryCosts.bBlocked = false;
  int closest_path_i = path.size();
  for(unsigned int k = 0; k < obj.predTrajectories.size(); k++)
  {
    for(unsigned int j = 0; j < obj.predTrajectories.at(k).size(); j++)
    {
      for(unsigned int i = 0; i < path.size(); i++)
      {
        //if(path.at(i).timeCost > -1)
        {
          double collision_distance = hypot(path.at(i).pos.x-obj.predTrajectories.at(k).at(j).pos.x, path.at(i).pos.y-obj.predTrajectories.at(k).at(j).pos.y);
          double collision_t = fabs(path.at(i).timeCost - obj.predTrajectories.at(k).at(j).timeCost);

          //if(collision_distance <= c_lateral_d && i < closest_path_i && collision_t < m_CollisionTimeDiff)
          if(collision_distance <= c_lateral_d && i < closest_path_i)
          {

            closest_path_i = i;
            double a = UtilityHNS::UtilityH::AngleBetweenTwoAnglesPositive(path.at(i).pos.a, obj.predTrajectories.at(k).at(j).pos.a)/M_PI;
            if(a < 0.25 && (currPose.v - obj.center.v) > 0)
              trajectoryCosts.closest_obj_velocity = (currPose.v - obj.center.v);
            else
              trajectoryCosts.closest_obj_velocity = currPose.v;

            collisionPoint = path.at(i);
            collisionPoint.collisionCost = collision_t;
            collisionPoint.cost = collision_distance;
            trajectoryCosts.bBlocked = true;
          }
        }
      }
    }
  }
}

int TrajectoryDynamicCosts::GetCurrentRollOutIndex(const std::vector<WayPoint>& path, const WayPoint& currState, const PlanningParams& params)
{
  RelativeInfo obj_info;
  PlanningHelpers::GetRelativeInfo(path, currState, obj_info);
  int currIndex = params.rollOutNumber/2 + floor(obj_info.perp_distance/params.rollOutDensity);
  if(currIndex < 0)
    currIndex = 0;
  else if(currIndex > params.rollOutNumber)
    currIndex = params.rollOutNumber;

  return currIndex;
}

void TrajectoryDynamicCosts::InitializeCosts(const vector<vector<WayPoint> >& rollOuts, const PlanningParams& params)
{
  m_TrajectoryCosts.clear();
  if(rollOuts.size()>0)
  {
    TrajectoryCost tc;
    int centralIndex = params.rollOutNumber/2;
    tc.lane_index = 0;
    for(unsigned int it=0; it< rollOuts.size(); it++)
    {
      tc.index = it;
      tc.relative_index = it - centralIndex;
      tc.distance_from_center = params.rollOutDensity*tc.relative_index;
      tc.priority_cost = fabs(tc.distance_from_center);
      tc.closest_obj_distance = params.horizonDistance;
      if(rollOuts.at(it).size() > 0)
        tc.lane_change_cost = rollOuts.at(it).at(0).laneChangeCost;
      m_TrajectoryCosts.push_back(tc);
    }
  }
}

void TrajectoryDynamicCosts::InitializeSafetyPolygon(const WayPoint& currState, const CAR_BASIC_INFO& carInfo, const VehicleState& vehicleState, const double& c_lateral_d, const double& c_long_front_d, const double& c_long_back_d)
{
  PlannerHNS::Mat3 invRotationMat(currState.pos.a-M_PI_2);
  PlannerHNS::Mat3 invTranslationMat(currState.pos.x, currState.pos.y);

  double corner_slide_distance = c_lateral_d/2.0;
  double ratio_to_angle = corner_slide_distance/carInfo.max_steer_angle;
  double slide_distance = vehicleState.steer * ratio_to_angle;

  GPSPoint bottom_left(-c_lateral_d ,-c_long_back_d,  currState.pos.z, 0);
  GPSPoint bottom_right(c_lateral_d, -c_long_back_d,  currState.pos.z, 0);

  GPSPoint top_right_car(c_lateral_d, carInfo.wheel_base/3.0 + carInfo.length/3.0,  currState.pos.z, 0);
  GPSPoint top_left_car(-c_lateral_d, carInfo.wheel_base/3.0 + carInfo.length/3.0, currState.pos.z, 0);

  GPSPoint top_right(c_lateral_d - slide_distance, c_long_front_d,  currState.pos.z, 0);
  GPSPoint top_left(-c_lateral_d - slide_distance , c_long_front_d, currState.pos.z, 0);

  bottom_left = invRotationMat*bottom_left;
  bottom_left = invTranslationMat*bottom_left;

  top_right = invRotationMat*top_right;
  top_right = invTranslationMat*top_right;

  bottom_right = invRotationMat*bottom_right;
  bottom_right = invTranslationMat*bottom_right;

  top_left = invRotationMat*top_left;
  top_left = invTranslationMat*top_left;

  top_right_car = invRotationMat*top_right_car;
  top_right_car = invTranslationMat*top_right_car;

  top_left_car = invRotationMat*top_left_car;
  top_left_car = invTranslationMat*top_left_car;

  m_SafetyBorder.points.clear();
  m_SafetyBorder.points.push_back(bottom_left) ;
  m_SafetyBorder.points.push_back(bottom_right) ;
  m_SafetyBorder.points.push_back(top_right_car) ;
  m_SafetyBorder.points.push_back(top_right) ;
  m_SafetyBorder.points.push_back(top_left) ;
  m_SafetyBorder.points.push_back(top_left_car) ;
}

void TrajectoryDynamicCosts::CalculateLateralAndLongitudinalCostsDynamic(const std::vector<PlannerHNS::DetectedObject>& obj_list, const vector<vector<WayPoint> >& rollOuts, const vector<WayPoint>& totalPaths,
    const WayPoint& currState, const PlanningParams& params, const CAR_BASIC_INFO& carInfo,
    const VehicleState& vehicleState, const double& c_lateral_d, const double& c_long_front_d, const double& c_long_back_d )
{

  RelativeInfo car_info;
  PlanningHelpers::GetRelativeInfo(totalPaths, currState, car_info);
  m_CollisionPoints.clear();

  for(unsigned int i=0; i < obj_list.size(); i++)
  {
    if(obj_list.at(i).label.compare("curb") == 0)
    {
      double d = hypot(obj_list.at(i).center.pos.y - currState.pos.y ,  obj_list.at(i).center.pos.x - currState.pos.x);
      if(d > params.minFollowingDistance + c_lateral_d)
        continue;
    }

    if(obj_list.at(i).bVelocity && obj_list.at(i).predTrajectories.size() > 0) // dynamic
    {

      for(unsigned int ir=0; ir < rollOuts.size(); ir++)
      {
        WayPoint collisionPoint;
        TrajectoryCost trajectoryCosts;
        CalculateIntersectionVelocities(rollOuts.at(ir), obj_list.at(i), currState, carInfo, c_lateral_d, collisionPoint,trajectoryCosts);
        if(trajectoryCosts.bBlocked)
        {
          RelativeInfo col_info;
          PlanningHelpers::GetRelativeInfo(totalPaths, collisionPoint, col_info);
          double longitudinalDist = PlanningHelpers::GetExactDistanceOnTrajectory(totalPaths, car_info, col_info);

          if(col_info.iFront == 0 && longitudinalDist > 0)
            longitudinalDist = -longitudinalDist;

          if(longitudinalDist < -carInfo.length || longitudinalDist > params.minFollowingDistance || fabs(longitudinalDist) < carInfo.width/2.0)
            continue;

          //std::cout << "LongDistance: " << longitudinalDist << std::endl;

          if(longitudinalDist >= -c_long_front_d && longitudinalDist < m_TrajectoryCosts.at(ir).closest_obj_distance)
            m_TrajectoryCosts.at(ir).closest_obj_distance = longitudinalDist;

          m_TrajectoryCosts.at(ir).closest_obj_velocity = trajectoryCosts.closest_obj_velocity;
          m_TrajectoryCosts.at(ir).bBlocked = true;

          m_CollisionPoints.push_back(collisionPoint);
        }
      }
    }
    else
    {
      RelativeInfo obj_info;
      WayPoint corner_p;
      for(unsigned int icon = 0; icon < obj_list.at(i).contour.size(); icon++)
      {
        if(m_SafetyBorder.PointInsidePolygon(m_SafetyBorder, obj_list.at(i).contour.at(icon)) == true)
        {
          for(unsigned int it=0; it< rollOuts.size(); it++)
            m_TrajectoryCosts.at(it).bBlocked = true;

          return;
        }

        corner_p.pos = obj_list.at(i).contour.at(icon);
        PlanningHelpers::GetRelativeInfo(totalPaths, corner_p, obj_info);
        double longitudinalDist = PlanningHelpers::GetExactDistanceOnTrajectory(totalPaths, car_info, obj_info);
        if(obj_info.iFront == 0 && longitudinalDist > 0)
          longitudinalDist = -longitudinalDist;


        if(longitudinalDist < -carInfo.length || longitudinalDist > params.minFollowingDistance)
          continue;

        longitudinalDist = longitudinalDist - c_long_front_d;

        for(unsigned int it=0; it< rollOuts.size(); it++)
        {
          double lateralDist = fabs(obj_info.perp_distance - m_TrajectoryCosts.at(it).distance_from_center);

          if(lateralDist > m_LateralSkipDistance)
            continue;

          if(lateralDist <= c_lateral_d && longitudinalDist > -carInfo.length && longitudinalDist < params.minFollowingDistance)
          {
            m_TrajectoryCosts.at(it).bBlocked = true;
            m_CollisionPoints.push_back(obj_info.perp_point);
          }

          if(lateralDist != 0)
            m_TrajectoryCosts.at(it).lateral_cost += 1.0/lateralDist;

          if(longitudinalDist != 0)
            m_TrajectoryCosts.at(it).longitudinal_cost += 1.0/fabs(longitudinalDist);

          if(longitudinalDist >= -c_long_front_d && longitudinalDist < m_TrajectoryCosts.at(it).closest_obj_distance)
          {
            m_TrajectoryCosts.at(it).closest_obj_distance = longitudinalDist;
            m_TrajectoryCosts.at(it).closest_obj_velocity = obj_list.at(i).center.v;
          }
        }
      }

    }
  }
}

double TrajectoryDynamicCosts::CalculateTurnAngle(const std::vector<WayPoint>& path, const WayPoint& currState, int distance)
{
  RelativeInfo car_info;
  PlanningHelpers::GetRelativeInfo(path, currState, car_info);
  GPSPoint rollout_start_pose = path.at(std::min(car_info.iFront + 3, int(path.size())-1)).pos;
  GPSPoint turn_pose = path.at(std::min(car_info.iFront + distance, int(path.size())-1)).pos;

  // std::cout << "ego x: " << rollout_start_pose.x << ", y: " << rollout_start_pose.y << std::endl;
  // std::cout << "turn x: " << turn_pose.x << ", y: " << turn_pose.y << std::endl;

  double m_turnAngle = 0;
  double hypot_length = hypot((rollout_start_pose.x - turn_pose.x), (rollout_start_pose.y - turn_pose.y));

  if(hypot_length <= 0.1)
    m_turnAngle = 0;
  else
    m_turnAngle = acos(abs(turn_pose.x - rollout_start_pose.x)/hypot_length)*180.0/3.14;
  if((turn_pose.y - rollout_start_pose.y) < 0)
    m_turnAngle = -1 * m_turnAngle;

    return m_turnAngle;
}

}
