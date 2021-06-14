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
#define SPIN_PROFILING

int main(int argc, char** argv)
{
  ros::init(argc, argv, "imm_ukf_pda_tracker");
  ImmUkfPda app;
  app.run();
  #ifndef SPIN_PROFILING
  //ros::spin();
  #endif
  #ifdef SPIN_PROFILING
  std::string print_file_path = std::getenv("HOME");
  print_file_path.append("/Documents/spin_profiling/imm_ukf_pda_track.csv");
  FILE *fp;
  fp = fopen(print_file_path.c_str(), "a");
  while(ros::ok()){
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    ros::spinOnce();
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    fprintf(fp, "%lld.%.9ld,%lld.%.9ld,%d\n",start_time.tv_sec,start_time.tv_nsec,end_time.tv_sec,end_time.tv_nsec,getpid());    
    fflush(fp);
  }  
  fclose(fp);
  #endif

  return 0;
}
