
/// \file DecisionMaker.cpp
/// \brief Initialize behaviors state machine, and calculate required parameters for the state machine transition conditions
/// \author Hatem Darweesh
/// \date Dec 14, 2016


#include "op_planner/DecisionMaker.h"
#include "op_utility/UtilityH.h"
#include "op_planner/PlanningHelpers.h"
#include "op_planner/MappingHelpers.h"
#include "op_planner/MatrixOperations.h"


namespace PlannerHNS
{

DecisionMaker::DecisionMaker()
{
  m_iCurrentTotalPathId = 0;
  pLane = 0;
  m_pCurrentBehaviorState = 0;
  m_pGoToGoalState = 0;
  m_pStopState= 0;
  m_pWaitState= 0;
  m_pMissionCompleteState= 0;
  m_pAvoidObstacleState = 0;
  m_pTrafficLightStopState = 0;
  m_pTrafficLightWaitState = 0;
  m_pStopSignStopState = 0;
  m_pStopSignWaitState = 0;
  m_pFollowState = 0;
  m_MaxLaneSearchDistance = 3.0;
  m_pStopState = 0;
  m_pMissionCompleteState = 0;
  m_pGoalState = 0;
  m_pGoToGoalState = 0;
  m_pWaitState = 0;
  m_pInitState = 0;
  m_pFollowState = 0;
  m_pAvoidObstacleState = 0;
  m_pPedestrianState = 0;
  m_sprintSpeed = -1;
  m_remainObstacleWaitingTime = 0;
  curveSlowDownCount = 400;
}

DecisionMaker::~DecisionMaker()
{
  delete m_pStopState;
  delete m_pMissionCompleteState;
  delete m_pGoalState;
  delete m_pGoToGoalState;
  delete m_pWaitState;
  delete m_pInitState;
  delete m_pFollowState;
  delete m_pAvoidObstacleState;
  delete m_pTrafficLightStopState;
  delete m_pTrafficLightWaitState;
  delete m_pStopSignWaitState;
  delete m_pStopSignStopState;
  delete m_pPedestrianState;
}

void DecisionMaker::Init(const ControllerParams& ctrlParams, const PlannerHNS::PlanningParams& params,const CAR_BASIC_INFO& carInfo, const double sprintSpeed)
   {
    m_CarInfo = carInfo;
    m_ControlParams = ctrlParams;
    m_params = params;
    m_sprintSpeed = sprintSpeed;

    m_pidVelocity.Init(0.01, 0.004, 0.01);
    m_pidVelocity.Setlimit(m_params.maxSpeed, 0);

    m_pidSprintVelocity.Init(0.01, 0.004, 0.01);
    m_pidSprintVelocity.Setlimit(sprintSpeed, 0);

    m_pidIntersectionVelocity.Init(0.01, 0.004, 0.01);
    m_pidIntersectionVelocity.Setlimit(m_params.maxSpeed*0.7, 0);

    m_pidStopping.Init(0.05, 0.05, 0.1);
    m_pidStopping.Setlimit(m_params.horizonDistance, 0);

    m_pidFollowing.Init(0.05, 0.05, 0.01);
    m_pidFollowing.Setlimit(m_params.minFollowingDistance, 0);

    m_prevTrafficLightID = -1;
    m_prevTrafficLightSignal = UNKNOWN_LIGHT;
    m_remainTrafficLightWaitingTime = 0;

    InitBehaviorStates();

    if(m_pCurrentBehaviorState)
      m_pCurrentBehaviorState->SetBehaviorsParams(&m_params);
   }

void DecisionMaker::InitBehaviorStates()
{

  m_pStopState         = new StopState(&m_params, 0, 0);
  m_pMissionCompleteState   = new MissionAccomplishedStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), 0);
  m_pGoalState        = new GoalStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pMissionCompleteState);
  m_pGoToGoalState       = new ForwardStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pGoalState);
  m_pInitState         = new InitStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pGoToGoalState);

  m_pFollowState        = new FollowStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pGoToGoalState);
  m_pAvoidObstacleState    = new SwerveStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pGoToGoalState);
  m_pStopSignWaitState    = new StopSignWaitStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pGoToGoalState);
  m_pStopSignStopState    = new StopSignStopStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pStopSignWaitState);

  m_pTrafficLightWaitState  = new TrafficLightWaitStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pGoToGoalState);
  m_pTrafficLightStopState  = new TrafficLightStopStateII(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pGoToGoalState);

  // Added by PHY
  m_pPedestrianState         = new PedestrianState(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pGoToGoalState);

  m_pIntersectionState         = new IntersectionState(m_pStopState->m_pParams, m_pStopState->GetCalcParams(), m_pGoToGoalState);

  m_pGoToGoalState->InsertNextState(m_pAvoidObstacleState);
  m_pGoToGoalState->InsertNextState(m_pStopSignStopState);
  m_pGoToGoalState->InsertNextState(m_pTrafficLightStopState);
  m_pGoToGoalState->InsertNextState(m_pFollowState);
  m_pGoToGoalState->InsertNextState(m_pPedestrianState);
  m_pGoToGoalState->InsertNextState(m_pIntersectionState);
  m_pGoToGoalState->decisionMakingCount = 0;//m_params.nReliableCount;

  m_pGoalState->InsertNextState(m_pGoToGoalState);

  m_pStopSignWaitState->decisionMakingTime = m_params.stopSignStopTime;
  m_pStopSignWaitState->InsertNextState(m_pStopSignStopState);
  m_pStopSignWaitState->InsertNextState(m_pGoalState);
  m_pStopSignWaitState->InsertNextState(m_pPedestrianState);

  m_pStopSignStopState->InsertNextState(m_pPedestrianState);

  m_pTrafficLightStopState->InsertNextState(m_pTrafficLightWaitState);
  m_pTrafficLightStopState->InsertNextState(m_pPedestrianState);

  m_pTrafficLightWaitState->InsertNextState(m_pTrafficLightStopState);
  m_pTrafficLightWaitState->InsertNextState(m_pGoalState);
  m_pTrafficLightWaitState->InsertNextState(m_pPedestrianState);

  m_pFollowState->InsertNextState(m_pAvoidObstacleState);
  m_pFollowState->InsertNextState(m_pStopSignStopState);
  m_pFollowState->InsertNextState(m_pTrafficLightStopState);
  m_pFollowState->InsertNextState(m_pGoalState);
  m_pFollowState->InsertNextState(m_pPedestrianState);
  m_pFollowState->decisionMakingCount = 0;//m_params.nReliableCount;

  m_pAvoidObstacleState->InsertNextState(m_pPedestrianState);

  m_pInitState->decisionMakingCount = 0;//m_params.nReliableCount;

  m_pCurrentBehaviorState = m_pInitState;
}

 bool DecisionMaker::GetNextTrafficLight(const int& prevTrafficLightId, const std::vector<PlannerHNS::TrafficLight>& trafficLights, PlannerHNS::TrafficLight& trafficL)
 {
   for(unsigned int i = 0; i < trafficLights.size(); i++)
   {
     double d = hypot(trafficLights.at(i).pos.y - state.pos.y, trafficLights.at(i).pos.x - state.pos.x);
     if(d <= trafficLights.at(i).stoppingDistance)
     {
       double a_diff = UtilityHNS::UtilityH::AngleBetweenTwoAnglesPositive(UtilityHNS::UtilityH::FixNegativeAngle(trafficLights.at(i).pos.a) , UtilityHNS::UtilityH::FixNegativeAngle(state.pos.a));

       if(a_diff < M_PI_2 && trafficLights.at(i).id != prevTrafficLightId)
       {
         //std::cout << "Detected Light, ID = " << trafficLights.at(i).id << ", Distance = " << d << ", Angle = " << trafficLights.at(i).pos.a*RAD2DEG << ", Car Heading = " << state.pos.a*RAD2DEG << ", Diff = " << a_diff*RAD2DEG << std::endl;
         trafficL = trafficLights.at(i);
         return true;
       }
     }
   }

   return false;
 }

 void DecisionMaker::CalculateImportantParameterForDecisionMaking(const PlannerHNS::VehicleState& car_state,
     const int& goalID, const bool& bEmergencyStop, const std::vector<TrafficLight>& detectedLights,
     const TrajectoryCost& bestTrajectory)
 {
   if(m_TotalPath.size() == 0) return;

   PreCalculatedConditions* pValues = m_pCurrentBehaviorState->GetCalcParams();

   if(m_CarInfo.max_deceleration != 0)
     pValues->minStoppingDistance = -pow(car_state.speed, 2)/(m_CarInfo.max_deceleration);

   pValues->iCentralTrajectory    = m_pCurrentBehaviorState->m_pParams->rollOutNumber/2;

  if(pValues->iPrevSafeTrajectory < 0)
    pValues->iPrevSafeTrajectory = pValues->iCentralTrajectory;

   pValues->stoppingDistances.clear();
   pValues->currentVelocity     = car_state.speed;
   pValues->bTrafficIsRed       = false;
   pValues->currentTrafficLightID   = -1;
   pValues->bFullyBlock       = false;

   pValues->distanceToNext = bestTrajectory.closest_obj_distance;
   pValues->velocityOfNext = bestTrajectory.closest_obj_velocity;

   if(bestTrajectory.index >=0 &&  bestTrajectory.index < (int)m_RollOuts.size())
     pValues->iCurrSafeTrajectory = bestTrajectory.index;
   else
     pValues->iCurrSafeTrajectory = pValues->iCentralTrajectory;

  pValues->bFullyBlock = bestTrajectory.bBlocked;

   if(bestTrajectory.lane_index >=0)
     pValues->iCurrSafeLane = bestTrajectory.lane_index;
   else
   {
     PlannerHNS::RelativeInfo info;
     PlannerHNS::PlanningHelpers::GetRelativeInfoRange(m_TotalPath, state, m_params.rollOutDensity*m_params.rollOutNumber/2.0 + 0.1, info);
     pValues->iCurrSafeLane = info.iGlobalPath;
   }

   double critical_long_front_distance =  m_CarInfo.wheel_base/2.0 + m_CarInfo.length/2.0 + m_params.verticalSafetyDistance;

  // HJW modified
  // ISSUE: Stop distnace should be calculated dynamically on real vehicle
  //if(ReachEndOfGlobalPath(pValues->minStoppingDistance + critical_long_front_distance, pValues->iCurrSafeLane))
  if(ReachEndOfGlobalPath(0.3, pValues->iCurrSafeLane))
    pValues->currentGoalID = -1;
  else
    pValues->currentGoalID = goalID;

  m_iCurrentTotalPathId = pValues->iCurrSafeLane;

  // For Intersection
  m_params.isInsideIntersection = m_isInsideIntersection;
  m_params.obstacleinRiskyArea = (m_riskyLeft || m_riskyRight);
  m_params.closestIntersectionDistance = m_closestIntersectionDistance;

  // std::cout << "isIn : " << m_params.isInsideIntersection << " left : " << m_params.turnLeft << " right : " << m_params.turnRight << std::endl;

  // For Traffic Signal

  int stopLineID = -1;
  int stopSignID = -1;
  double stopLineLength = -1;
  int trafficLightID = -1;
  double distanceToClosestStopLine = 0;
  bool bShouldForward = true;
  bool traffic_detected = false;
  double remain_time = 0;

  distanceToClosestStopLine = PlanningHelpers::CalculateStopLineDistance_RUBIS(m_TotalPath.at(pValues->iCurrSafeLane), state, m_Map.stopLines, stopLineID, stopLineLength, trafficLightID) - critical_long_front_distance;

  // std::cout << "StopLineID : " << stopLineID << ", TrafficLightID : " << trafficLightID << ", Distance: " << distanceToClosestStopLine << ", MinStopDistance: " << pValues->minStoppingDistance << std::endl;
  // std::cout << "detected Lights # : " << detectedLights.size() << std::endl;

  if(m_pCurrentBehaviorState->m_pParams->enableTrafficLightBehavior){

    for(unsigned int i=0; i< detectedLights.size(); i++)
    {
      if(detectedLights.at(i).id == trafficLightID){
        traffic_detected = true;

        ////// V2X without remain time
        remain_time = m_remainTrafficLightWaitingTime;
        // new light case
        if(detectedLights.at(i).id != m_prevTrafficLightID){
          if(detectedLights.at(i).lightState == GREEN_LIGHT){
            remain_time = detectedLights.at(i).routine.at(1); // Add yellow light
          }
          else{ // For yellow, red case
            remain_time = 0;
          }
        }
        else if(detectedLights.at(i).lightState != m_prevTrafficLightSignal){
          if(detectedLights.at(i).lightState == GREEN_LIGHT){ // R -> G
            // G Time + Y Time
            remain_time = detectedLights.at(i).routine.at(0) + detectedLights.at(i).routine.at(1);
          }
          else if(detectedLights.at(i).lightState == YELLOW_LIGHT){ // G -> Y
            remain_time = detectedLights.at(i).routine.at(1);
          }
          else{ // Y -> R
            remain_time = detectedLights.at(i).routine.at(2);
          }
        }
        else{
          remain_time -= 0.01; // spin period;
          if(remain_time < 0) remain_time = 0.0;
        }

        if(distanceToClosestStopLine < m_params.stopLineDetectionDistance && distanceToClosestStopLine > 0){
          bool bGreenTrafficLight = !(detectedLights.at(i).lightState == RED_LIGHT);
          double reachableDistance = m_params.maxSpeed * detectedLights.at(i).remainTime / 2;
          bShouldForward = (bGreenTrafficLight && reachableDistance > distanceToClosestStopLine) ||
                        (!bGreenTrafficLight && reachableDistance < distanceToClosestStopLine);

          pValues->currentTrafficLightID = trafficLightID;
          pValues->stoppingDistances.push_back(distanceToClosestStopLine);
        }

        m_prevTrafficLightID = trafficLightID;
        m_prevTrafficLightSignal = detectedLights.at(i).lightState;

        ////// V2X with remain time and stop line length
        // double remain_time = detectedLights.at(i).remainTime;
        // if(detectedLights.at(i).lightState == GREEN_LIGHT){ // Add time for yellow lights time
        //   remain_time += detectedLights.at(i).routine.at(1); // Add yellow light
        // }
        // double reachableDistance = m_params.maxSpeed * detectedLights.at(i).remainTime / 2;
        // bool bGreenTrafficLight = !(detectedLights.at(i).lightState == RED_LIGHT);
        // bShouldForward = (bGreenTrafficLight && reachableDistance > distanceToClosestStopLine + stopLineLength) ||
        //               (!bGreenTrafficLight && reachableDistance < distanceToClosestStopLine);

        ////// V2X with remain time without stop line length
        // double remain_time = detectedLights.at(i).remainTime;
        // if(detectedLights.at(i).lightState == GREEN_LIGHT){ // Add time for yellow lights time
        //   remain_time += detectedLights.at(i).routine.at(1); // Add yellow light
        // }
        // double reachableDistance = m_params.maxSpeed * detectedLights.at(i).remainTime / 2;
        // bool bGreenTrafficLight = !(detectedLights.at(i).lightState == RED_LIGHT);
        // bShouldForward = (bGreenTrafficLight && reachableDistance > distanceToClosestStopLine) ||
        //               (!bGreenTrafficLight && reachableDistance < distanceToClosestStopLine);
      }
    }

    if(!traffic_detected){
      m_prevTrafficLightID = -1;
      m_prevTrafficLightSignal = UNKNOWN_LIGHT;
    }

    m_remainTrafficLightWaitingTime = remain_time;
  }

   pValues->bTrafficIsRed = !bShouldForward;

  if(bEmergencyStop)
  {
    pValues->bFullyBlock = true;
    pValues->distanceToNext = 1;
    pValues->velocityOfNext = 0;
  }
   //cout << "Distances: " << pValues->stoppingDistances.size() << ", Distance To Stop : " << pValues->distanceToStop << endl;
 }

 void DecisionMaker::UpdateCurrentLane(const double& search_distance)
 {
   PlannerHNS::Lane* pMapLane = 0;
  PlannerHNS::Lane* pPathLane = 0;
  pPathLane = MappingHelpers::GetLaneFromPath(state, m_TotalPath.at(m_iCurrentTotalPathId));
  if(!pPathLane)
  {
    std::cout << "Performance Alert: Can't Find Lane Information in Global Path, Searching the Map :( " << std::endl;
    pMapLane  = MappingHelpers::GetClosestLaneFromMap(state, m_Map, search_distance);
  }

  if(pPathLane)
    pLane = pPathLane;
  else if(pMapLane)
    pLane = pMapLane;
  else
    pLane = 0;
 }

 bool DecisionMaker::ReachEndOfGlobalPath(const double& min_distance, const int& iGlobalPathIndex)
 {
   if(m_TotalPath.size()==0) return false;

   PlannerHNS::RelativeInfo info;
   PlanningHelpers::GetRelativeInfo(m_TotalPath.at(iGlobalPathIndex), state, info);

   double d = 0;
   for(unsigned int i = info.iFront; i < m_TotalPath.at(iGlobalPathIndex).size()-1; i++)
   {
     d+= hypot(m_TotalPath.at(iGlobalPathIndex).at(i+1).pos.y - m_TotalPath.at(iGlobalPathIndex).at(i).pos.y, m_TotalPath.at(iGlobalPathIndex).at(i+1).pos.x - m_TotalPath.at(iGlobalPathIndex).at(i).pos.x);
     if(d > min_distance)
       return false;
   }

   return true;
 }

void DecisionMaker::SetNewGlobalPath(const std::vector<std::vector<WayPoint> >& globalPath)
{
  if(m_pCurrentBehaviorState)
  {
    m_pCurrentBehaviorState->GetCalcParams()->bNewGlobalPath = true;
    m_TotalOriginalPath = globalPath;
  }
 }

bool DecisionMaker::SelectSafeTrajectory()
{
  bool bNewTrajectory = false;
  PlannerHNS::PreCalculatedConditions *preCalcPrams = m_pCurrentBehaviorState->GetCalcParams();

  if(!preCalcPrams || m_RollOuts.size() == 0) return bNewTrajectory;

  int currIndex = PlannerHNS::PlanningHelpers::GetClosestNextPointIndexFast(m_Path, state);
  int index_limit = 0;
  if(index_limit<=0)
    index_limit =  m_Path.size()/2.0;
  if(currIndex > index_limit
      || preCalcPrams->bRePlan
      || preCalcPrams->bNewGlobalPath)
  {
    std::cout << "New Local Plan !! " << currIndex << ", "<< preCalcPrams->bRePlan << ", " << preCalcPrams->bNewGlobalPath  << ", " <<  m_TotalOriginalPath.at(0).size() << ", PrevLocal: " << m_Path.size();
    m_Path = m_RollOuts.at(preCalcPrams->iCurrSafeTrajectory);
    std::cout << ", NewLocal: " << m_Path.size() << std::endl;

    preCalcPrams->bNewGlobalPath = false;
    preCalcPrams->bRePlan = false;
    bNewTrajectory = true;
  }

  return bNewTrajectory;
 }

 PlannerHNS::BehaviorState DecisionMaker::GenerateBehaviorState(const PlannerHNS::VehicleState& vehicleState)
 {
  PlannerHNS::PreCalculatedConditions *preCalcPrams = m_pCurrentBehaviorState->GetCalcParams();

  m_pCurrentBehaviorState = m_pCurrentBehaviorState->GetNextState();
  if(m_pCurrentBehaviorState==0)
    m_pCurrentBehaviorState = m_pInitState;

  PlannerHNS::BehaviorState currentBehavior;

  currentBehavior.state = m_pCurrentBehaviorState->m_Behavior;
  currentBehavior.followDistance = preCalcPrams->distanceToNext;

  currentBehavior.minVelocity    = 0;
  currentBehavior.stopDistance   = preCalcPrams->distanceToStop();
  currentBehavior.followVelocity   = preCalcPrams->velocityOfNext;
  if(preCalcPrams->iPrevSafeTrajectory<0 || preCalcPrams->iPrevSafeTrajectory >= m_RollOuts.size())
    currentBehavior.iTrajectory    = preCalcPrams->iCurrSafeTrajectory;
  else
    currentBehavior.iTrajectory    = preCalcPrams->iPrevSafeTrajectory;

  double average_braking_distance = -pow(vehicleState.speed, 2)/(m_CarInfo.max_deceleration) + m_params.additionalBrakingDistance;

  if(average_braking_distance  < m_params.minIndicationDistance)
    average_braking_distance = m_params.minIndicationDistance;

  double minDistanceToRollOut = 0;
  for(int i=0; i<m_RollOuts.size(); i++){
    const PlannerHNS::WayPoint rollout_start_waypoint = m_RollOuts.at(i).at(std::min(10, int(m_RollOuts.at(i).size()))-1);

    double direct_distance = hypot(rollout_start_waypoint.pos.y - state.pos.y, rollout_start_waypoint.pos.x - state.pos.x);

    if(minDistanceToRollOut == 0 || minDistanceToRollOut > direct_distance){
      minDistanceToRollOut = direct_distance;
      currentBehavior.currTrajectory = i;
    }
  }

  // Check turn
  // Detects whether or not to turning 50m ahead
  m_turnWaypoint = m_RollOuts.at(currentBehavior.currTrajectory).at(std::min(100, int(m_RollOuts.at(currentBehavior.currTrajectory).size()))-1);
  
  currentBehavior.indicator = PlanningHelpers::GetIndicatorsFromPath(m_Path, state, average_braking_distance );

  return currentBehavior;
 }

 double DecisionMaker::UpdateVelocityDirectlyToTrajectory(const BehaviorState& beh, const VehicleState& CurrStatus, const double& dt)
 {
  if(m_TotalOriginalPath.size() ==0 ) return 0;

  RelativeInfo info, total_info;
  PlanningHelpers::GetRelativeInfo(m_TotalOriginalPath.at(m_iCurrentTotalPathId), state, total_info);
  PlanningHelpers::GetRelativeInfo(m_Path, state, info);
  double average_braking_distance = -pow(CurrStatus.speed, 2)/(m_CarInfo.max_deceleration) + m_params.additionalBrakingDistance;
  double max_velocity  = PlannerHNS::PlanningHelpers::GetVelocityAhead(m_TotalOriginalPath.at(m_iCurrentTotalPathId), total_info, total_info.iBack, average_braking_distance);

  unsigned int point_index = 0;
  double critical_long_front_distance = m_CarInfo.length/2.0;

  if(beh.state == PEDESTRIAN_STATE){
    double desiredVelocity = -1;
    return desiredVelocity;
  }
  else if(beh.state == INTERSECTION_STATE){
    
    max_velocity = m_params.maxSpeed*0.7;
    double target_velocity = max_velocity;

    double e = target_velocity - CurrStatus.speed;

    m_pidSprintVelocity.getPID(e);
    m_pidVelocity.getPID(e);
    double desiredVelocity = m_pidIntersectionVelocity.getPID(e);
    
    if(desiredVelocity > max_velocity)
      desiredVelocity = max_velocity;
    else if(desiredVelocity < m_params.minSpeed)
      desiredVelocity = 0;

    // std::cout << "o_a : " << m_params.obstacleinRiskyArea << "f_d : " << beh.followDistance << std::endl;
    // std::cout << "remain_time : " << m_remainObstacleWaitingTime << std::endl;

    // Wait for predetermined time
    if(m_remainObstacleWaitingTime > 0){
      m_remainObstacleWaitingTime--;
      desiredVelocity = 0;
    }
    else if(m_params.obstacleinRiskyArea){
      // double desiredAcceleration = m_params.maxSpeed * m_params.maxSpeed / 2 / std::max(beh.stopDistance - m_params.stopLineMargin, 0.1);
      // double desiredVelocity = m_params.maxSpeed - desiredAcceleration * 0.1; // 0.1 stands for delta t.
      // if(desiredVelocity < 0.5)
      //   desiredVelocity = 0;
      desiredVelocity = 0;
      m_remainObstacleWaitingTime = int(m_obstacleWaitingTimeinIntersection * 100);
    }
    // else if(beh.followDistance < 30){
    //   desiredVelocity = 0;
    //   m_remainObstacleWaitingTime = int(m_obstacleWaitingTimeinIntersection * 100);
    // }

    for(unsigned int i = 0; i < m_Path.size(); i++)
      m_Path.at(i).v = desiredVelocity;

    return desiredVelocity;
  }
  else if(beh.state == TRAFFIC_LIGHT_STOP_STATE || beh.state == TRAFFIC_LIGHT_WAIT_STATE)
  {
    double desiredAcceleration = m_params.maxSpeed * m_params.maxSpeed / 2 / std::max(beh.stopDistance - m_params.stopLineMargin, 0.1);
    double desiredVelocity = m_params.maxSpeed - desiredAcceleration * 0.1; // 0.1 stands for delta t.
    
    double e = max_velocity - CurrStatus.speed;
    m_pidSprintVelocity.getPID(e);
    m_pidVelocity.getPID(e);
    m_pidIntersectionVelocity.getPID(e);

    if(desiredVelocity < 0.5)
      desiredVelocity = 0;

    for(unsigned int i = 0; i < m_Path.size(); i++)
      m_Path.at(i).v = desiredVelocity;
    return desiredVelocity;
  }
  else if(beh.state == STOP_SIGN_STOP_STATE)
  {
    PlanningHelpers::GetFollowPointOnTrajectory(m_Path, info, beh.stopDistance - critical_long_front_distance, point_index);

    double e = -beh.stopDistance;
    double desiredVelocity = m_pidStopping.getPID(e);

    double e2 = max_velocity - CurrStatus.speed;
    m_pidSprintVelocity.getPID(e2);
    m_pidVelocity.getPID(e2);
    m_pidIntersectionVelocity.getPID(e2);

//    std::cout << "Stopping : e=" << e << ", desiredPID=" << desiredVelocity << ", PID: " << m_pidStopping.ToString() << std::endl;

    if(desiredVelocity > max_velocity)
      desiredVelocity = max_velocity;
    else if(desiredVelocity < m_params.minSpeed)
      desiredVelocity = 0;

    for(unsigned int i =  0; i < m_Path.size(); i++)
      m_Path.at(i).v = desiredVelocity;

    return desiredVelocity;
  }
  else if(beh.state == STOP_SIGN_WAIT_STATE)
  {
    double target_velocity = 0;
    for(unsigned int i = 0; i < m_Path.size(); i++)
      m_Path.at(i).v = target_velocity;

    return target_velocity;
  }
  else if(beh.state == FOLLOW_STATE)
  {

    double deceleration_critical = 0;
    double inv_time = 2.0*((beh.followDistance- (critical_long_front_distance+m_params.additionalBrakingDistance))-CurrStatus.speed);
    if(inv_time <= 0)
      deceleration_critical = m_CarInfo.max_deceleration;
    else
      deceleration_critical = CurrStatus.speed*CurrStatus.speed/inv_time;

    if(deceleration_critical > 0) deceleration_critical = -deceleration_critical;
    if(deceleration_critical < - m_CarInfo.max_acceleration) deceleration_critical = - m_CarInfo.max_acceleration;

    double desiredVelocity = (deceleration_critical * dt) + CurrStatus.speed;

    if(desiredVelocity > m_params.maxSpeed)
      desiredVelocity = m_params.maxSpeed;

    if((desiredVelocity < 0.1 && desiredVelocity > -0.1) || beh.followDistance <= 0) //use only effective velocities
      desiredVelocity = 0;

    //std::cout << "Acc: V: " << desiredVelocity << ", Accel: " << deceleration_critical<< std::endl;

    for(unsigned int i = 0; i < m_Path.size(); i++)
      m_Path.at(i).v = desiredVelocity;

    return desiredVelocity;

  }
  else if(beh.state == FORWARD_STATE)
  {
    if(m_sprintSwitch == true){
      max_velocity = m_sprintSpeed;
    }

    double target_velocity = max_velocity;
    bool bSlowBecauseChange=false;

    // std::cout << "curr Traj : " << beh.currTrajectory << ", curr Safe Traj : " << m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory << std::endl;
    // if(m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory != m_pCurrentBehaviorState->GetCalcParams()->iCentralTrajectory)
    if(beh.currTrajectory != m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory)
    {
      // target_velocity /= fabs(beh.currTrajectory - m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory);
      target_velocity *= 0.5;
      bSlowBecauseChange = true;
    }
    
    double e = target_velocity - CurrStatus.speed;
    double desiredVelocity = 0; 

    double sprint_pid_vel, forward_pid_vel;
    sprint_pid_vel = m_pidSprintVelocity.getPID(e);
    forward_pid_vel = m_pidVelocity.getPID(e);
    m_pidIntersectionVelocity.getPID(e);

    if(m_sprintSwitch == true)
      desiredVelocity = sprint_pid_vel;
    else
      desiredVelocity = forward_pid_vel;

    if(m_params.enableSlowDownOnCurve){
      GPSPoint curr_point = m_Path.at(info.iFront).pos; // current waypoint (p1)
      GPSPoint near_point = m_Path.at(std::min(info.iFront + 3, int(m_Path.size())-1)).pos; // waypoint after 1.5m (p2)
      GPSPoint far_point = m_Path.at(std::min(info.iFront + 100, int(m_Path.size())-1)).pos; // waypoint afeter 50m (p3)

      double deg_1 = atan2((near_point.y - curr_point.y), (near_point.x - curr_point.x)) / 3.14 * 180;
      double deg_2 = atan2((far_point.y - curr_point.y), (far_point.x - curr_point.x)) / 3.14 * 180;
      double angle_diff = std::abs(deg_1 - deg_2); // angle between p1p2 and p1p3
      if (angle_diff > 180){
        angle_diff = 360 - angle_diff;
      }

      // std::cout << "curvature : " << angle_diff << std::endl;

      if (angle_diff > 7){ // Slow down when angle is large
        desiredVelocity = m_params.maxSpeed * 40 / (angle_diff + 33);
        // desiredVelocity = m_params.maxSpeed * 17 / (angle_diff + 10);
        if(desiredVelocity > previous_velocity){
          desiredVelocity = previous_velocity;
        }
        curveSlowDownCount = 0;
      }
      // if (angle_diff > 30){
      //   desiredVelocity = m_params.maxSpeed * 0.8;
      //   curveSlowDownCount = 0;
      //   // desiredVelocity = max_velocity * 100 / (angle_diff + 90);
      // }
      // else if(angle_diff > 10){
      //   desiredVelocity = m_params.maxSpeed * 0.6;
      //   curveSlowDownCount = 0;
      // }
      else if(curveSlowDownCount < 400){ // wait 4 sec when angle become less than 7 // TODO: Check its feasibility when pure pursuit is sufficiently tuned
        desiredVelocity = previous_velocity + (m_params.maxSpeed - previous_velocity) / 100;
        curveSlowDownCount += 1;
      }
      else{
        desiredVelocity = m_params.maxSpeed;
      }

      if(desiredVelocity < m_params.maxSpeed * m_params.curveVelocityRatio){ // minimum of target velocity is max_speed / 2
        desiredVelocity = m_params.maxSpeed * m_params.curveVelocityRatio; 
      }
      previous_velocity = desiredVelocity;
    }

    for(auto it = m_Path.begin(); it != m_Path.end(); ++it){
      (*it).v = desiredVelocity;
    }

    // std::cout << "desiredVelocity : " << desiredVelocity << std::endl;

    // std::cout << "Target Velocity: " << target_velocity << ", desired : " << desiredVelocity << ", Change Slowdown: " << bSlowBecauseChange  << std::endl;

    if(desiredVelocity>max_velocity)
      desiredVelocity = max_velocity;
    else if(desiredVelocity < m_params.minSpeed)
      desiredVelocity = 0;

    return desiredVelocity;
  }
  else if(beh.state == OBSTACLE_AVOIDANCE_STATE )
  {
    double target_velocity = max_velocity;
    bool bSlowBecauseChange=false;
    // std::cout << "curr Traj : " << beh.currTrajectory << ", curr Safe Traj : " << m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory << std::endl;
    // if(m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory != m_pCurrentBehaviorState->GetCalcParams()->iCentralTrajectory)
    if(beh.currTrajectory != m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory)
    {
      // target_velocity /= fabs(beh.currTrajectory - m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory);
      target_velocity *= 0.5;
      bSlowBecauseChange = true;
    }

    double e = target_velocity - CurrStatus.speed;
    double desiredVelocity = m_pidVelocity.getPID(e);
    m_pidSprintVelocity.getPID(e);    
    m_pidIntersectionVelocity.getPID(e);
    
    if(desiredVelocity>max_velocity)
      desiredVelocity = max_velocity;
    else if(desiredVelocity < m_params.minSpeed)
      desiredVelocity = 0;

    for(unsigned int i = 0; i < m_Path.size(); i++)
      m_Path.at(i).v = desiredVelocity;

    // std::cout << "Target Velocity: " << desiredVelocity << ", Change Slowdown: " << bSlowBecauseChange  << std::endl;

    return desiredVelocity;
  }
  else
  {
    double target_velocity = 0;
    for(unsigned int i = 0; i < m_Path.size(); i++)
      m_Path.at(i).v = target_velocity;

    return target_velocity;
  }

  return max_velocity;
 }

 PlannerHNS::BehaviorState DecisionMaker::DoOneStep(
    const double& dt,
    const PlannerHNS::WayPoint currPose,
    const PlannerHNS::VehicleState& vehicleState,
    const int& goalID,
    const std::vector<TrafficLight>& trafficLight,
    const TrajectoryCost& tc,
    const bool& bEmergencyStop)
{

   PlannerHNS::BehaviorState beh;
   state = currPose;
   m_TotalPath.clear();
  for(unsigned int i = 0; i < m_TotalOriginalPath.size(); i++)
  {
    t_centerTrajectorySmoothed.clear();
    PlannerHNS::PlanningHelpers::ExtractPartFromPointToDistanceDirectionFast(m_TotalOriginalPath.at(i), state, m_params.horizonDistance ,  m_params.pathDensity , t_centerTrajectorySmoothed);
    m_TotalPath.push_back(t_centerTrajectorySmoothed);
  }

  if(m_TotalPath.size()==0) return beh;

  UpdateCurrentLane(m_MaxLaneSearchDistance);

  CalculateImportantParameterForDecisionMaking(vehicleState, goalID, bEmergencyStop, trafficLight, tc);

  // Enable if left turn scenario needed
  // CheckTurn();

  // PrintTurn();

  beh = GenerateBehaviorState(vehicleState);

  beh.bNewPlan = SelectSafeTrajectory();

  beh.maxVelocity = UpdateVelocityDirectlyToTrajectory(beh, vehicleState, dt);
  
  //std::cout << "Eval_i: " << tc.index << ", Curr_i: " <<  m_pCurrentBehaviorState->GetCalcParams()->iCurrSafeTrajectory << ", Prev_i: " << m_pCurrentBehaviorState->GetCalcParams()->iPrevSafeTrajectory << std::endl;

  return beh;
 }

 // Added by PHY
  void DecisionMaker::UpdatePedestrianAppearence(const bool pedestrianAppearence){
    m_params.pedestrianAppearence = pedestrianAppearence;
  }

  void DecisionMaker::printPedestrianAppearence(){
    if(m_params.pedestrianAppearence == true){
      std::cout<<"Pedestrian Appearence: True"<<std::endl;  
    }
    else{
        std::cout<<"Pedestrian Appearence: False"<<std::endl;  
    }

  }

  void DecisionMaker::CheckTurn(){
    if(abs(m_turnAngle) > m_turnThreshold){
      if(m_turnAngle > 0){
        m_params.turnLeft = true;
        m_params.turnRight = false;
      }
      else{
        m_params.turnLeft = false;
        m_params.turnRight = true;
      }
    }
    else{
      m_params.turnLeft = false;
      m_params.turnRight = false;
    }
  }

  void DecisionMaker::PrintTurn(){
    std::cout<<"LEFT:"<<m_params.turnLeft<<std::endl;
    std::cout<<"RIGHT:"<<m_params.turnRight<<std::endl;
  }

  std::string DecisionMaker::ToString(STATE_TYPE beh)
  {
    std::string str = "Unknown";
    switch(beh)
    {
    case PlannerHNS::INITIAL_STATE:
      str = "Init";
      break;
    case PlannerHNS::EMERGENCY_STATE:
      str = "Emergency";
      break;
    case PlannerHNS::FORWARD_STATE:
      str = "Forward";
      break;
    case PlannerHNS::STOPPING_STATE:
      str = "Stop";
      break;
    case PlannerHNS::FINISH_STATE:
      str = "End";
      break;
    case PlannerHNS::FOLLOW_STATE:
      str = "Follow";
      break;
    case PlannerHNS::OBSTACLE_AVOIDANCE_STATE:
      str = "Swerving";
      break;
    case PlannerHNS::TRAFFIC_LIGHT_STOP_STATE:
      str = "Light Stop";
      break;
    case PlannerHNS::TRAFFIC_LIGHT_WAIT_STATE:
      str = "Light Wait";
      break;
    case PlannerHNS::STOP_SIGN_STOP_STATE:
      str = "Sign Stop";
      break;
    case PlannerHNS::STOP_SIGN_WAIT_STATE:
      str = "Sign Wait";
      break;
    case PlannerHNS::INTERSECTION_STATE:
      str = "Intersection";
      break;
    default:
      str = "Unknown";
      break;
    }

    return str;
  }
} /* namespace PlannerHNS */
