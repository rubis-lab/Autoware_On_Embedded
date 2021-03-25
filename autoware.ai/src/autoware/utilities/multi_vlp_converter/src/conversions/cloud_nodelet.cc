/*
 *  Copyright (C) 2012 Austin Robot Technology, Jack O'Quin
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** @file

    This ROS nodelet converts raw Velodyne 3D LIDAR packets to a
    PointCloud2.

*/

#include <ros/ros.h>
#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>

#include "multi_vlp_converter/convert.h"

namespace multi_vlp_converter
{
  class CloudNodelet: public nodelet::Nodelet
  {
  public:

    CloudNodelet() {}
    ~CloudNodelet() {}

  private:

    virtual void onInit();
    boost::shared_ptr<Convert> conv_;
  };

  /** @brief Nodelet initialization. */
  void CloudNodelet::onInit()
  {
    conv_.reset(new Convert(getNodeHandle(), getPrivateNodeHandle(), getName()));
  }

} // namespace multi_vlp_converter


// Register this plugin with pluginlib.  Names must match nodelets.xml.
//
// parameters: class type, base class type
PLUGINLIB_EXPORT_CLASS(multi_vlp_converter::CloudNodelet, nodelet::Nodelet)
