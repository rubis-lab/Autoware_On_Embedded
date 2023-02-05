/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <imm_ukf_pda/imm_ukf_pda.h>
#include <rubis_lib/sched.hpp>
#define SPIN_PROFILING

int main(int argc, char** argv)
{
  ros::init(argc, argv, "imm_ukf_pda_tracker");
  ros::NodeHandle private_nh("~");

  // Scheduling Setup
  int rate;
  private_nh.param<int>("/imm_ukf_pda_track/rate", rate, 10);

  ImmUkfPda app;
  app.run();

  ros::Rate r(rate);
  while(ros::ok()){
    rubis::start_task_profiling();

    ros::spinOnce();

    rubis::stop_task_profiling(rubis::instance_, 0);

    r.sleep();
  }

  return 0;
}
