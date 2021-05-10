// Auto-generated. Do not edit!

// (in-package autoware_config_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class ConfigDecisionMaker {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.auto_mission_reload = null;
      this.auto_engage = null;
      this.auto_mission_change = null;
      this.use_fms = null;
      this.disuse_vector_map = null;
      this.sim_mode = null;
      this.insert_stop_line_wp = null;
      this.num_of_steer_behind = null;
      this.change_threshold_dist = null;
      this.change_threshold_angle = null;
      this.goal_threshold_dist = null;
      this.goal_threshold_vel = null;
      this.stopped_vel = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('auto_mission_reload')) {
        this.auto_mission_reload = initObj.auto_mission_reload
      }
      else {
        this.auto_mission_reload = false;
      }
      if (initObj.hasOwnProperty('auto_engage')) {
        this.auto_engage = initObj.auto_engage
      }
      else {
        this.auto_engage = false;
      }
      if (initObj.hasOwnProperty('auto_mission_change')) {
        this.auto_mission_change = initObj.auto_mission_change
      }
      else {
        this.auto_mission_change = false;
      }
      if (initObj.hasOwnProperty('use_fms')) {
        this.use_fms = initObj.use_fms
      }
      else {
        this.use_fms = false;
      }
      if (initObj.hasOwnProperty('disuse_vector_map')) {
        this.disuse_vector_map = initObj.disuse_vector_map
      }
      else {
        this.disuse_vector_map = false;
      }
      if (initObj.hasOwnProperty('sim_mode')) {
        this.sim_mode = initObj.sim_mode
      }
      else {
        this.sim_mode = false;
      }
      if (initObj.hasOwnProperty('insert_stop_line_wp')) {
        this.insert_stop_line_wp = initObj.insert_stop_line_wp
      }
      else {
        this.insert_stop_line_wp = false;
      }
      if (initObj.hasOwnProperty('num_of_steer_behind')) {
        this.num_of_steer_behind = initObj.num_of_steer_behind
      }
      else {
        this.num_of_steer_behind = 0;
      }
      if (initObj.hasOwnProperty('change_threshold_dist')) {
        this.change_threshold_dist = initObj.change_threshold_dist
      }
      else {
        this.change_threshold_dist = 0.0;
      }
      if (initObj.hasOwnProperty('change_threshold_angle')) {
        this.change_threshold_angle = initObj.change_threshold_angle
      }
      else {
        this.change_threshold_angle = 0.0;
      }
      if (initObj.hasOwnProperty('goal_threshold_dist')) {
        this.goal_threshold_dist = initObj.goal_threshold_dist
      }
      else {
        this.goal_threshold_dist = 0.0;
      }
      if (initObj.hasOwnProperty('goal_threshold_vel')) {
        this.goal_threshold_vel = initObj.goal_threshold_vel
      }
      else {
        this.goal_threshold_vel = 0.0;
      }
      if (initObj.hasOwnProperty('stopped_vel')) {
        this.stopped_vel = initObj.stopped_vel
      }
      else {
        this.stopped_vel = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ConfigDecisionMaker
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [auto_mission_reload]
    bufferOffset = _serializer.bool(obj.auto_mission_reload, buffer, bufferOffset);
    // Serialize message field [auto_engage]
    bufferOffset = _serializer.bool(obj.auto_engage, buffer, bufferOffset);
    // Serialize message field [auto_mission_change]
    bufferOffset = _serializer.bool(obj.auto_mission_change, buffer, bufferOffset);
    // Serialize message field [use_fms]
    bufferOffset = _serializer.bool(obj.use_fms, buffer, bufferOffset);
    // Serialize message field [disuse_vector_map]
    bufferOffset = _serializer.bool(obj.disuse_vector_map, buffer, bufferOffset);
    // Serialize message field [sim_mode]
    bufferOffset = _serializer.bool(obj.sim_mode, buffer, bufferOffset);
    // Serialize message field [insert_stop_line_wp]
    bufferOffset = _serializer.bool(obj.insert_stop_line_wp, buffer, bufferOffset);
    // Serialize message field [num_of_steer_behind]
    bufferOffset = _serializer.uint32(obj.num_of_steer_behind, buffer, bufferOffset);
    // Serialize message field [change_threshold_dist]
    bufferOffset = _serializer.float64(obj.change_threshold_dist, buffer, bufferOffset);
    // Serialize message field [change_threshold_angle]
    bufferOffset = _serializer.float64(obj.change_threshold_angle, buffer, bufferOffset);
    // Serialize message field [goal_threshold_dist]
    bufferOffset = _serializer.float64(obj.goal_threshold_dist, buffer, bufferOffset);
    // Serialize message field [goal_threshold_vel]
    bufferOffset = _serializer.float64(obj.goal_threshold_vel, buffer, bufferOffset);
    // Serialize message field [stopped_vel]
    bufferOffset = _serializer.float64(obj.stopped_vel, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ConfigDecisionMaker
    let len;
    let data = new ConfigDecisionMaker(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [auto_mission_reload]
    data.auto_mission_reload = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [auto_engage]
    data.auto_engage = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [auto_mission_change]
    data.auto_mission_change = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [use_fms]
    data.use_fms = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [disuse_vector_map]
    data.disuse_vector_map = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [sim_mode]
    data.sim_mode = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [insert_stop_line_wp]
    data.insert_stop_line_wp = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [num_of_steer_behind]
    data.num_of_steer_behind = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [change_threshold_dist]
    data.change_threshold_dist = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [change_threshold_angle]
    data.change_threshold_angle = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [goal_threshold_dist]
    data.goal_threshold_dist = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [goal_threshold_vel]
    data.goal_threshold_vel = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [stopped_vel]
    data.stopped_vel = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 51;
  }

  static datatype() {
    // Returns string type for a message object
    return 'autoware_config_msgs/ConfigDecisionMaker';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '35bae6da1772c1676984cb320dd14d09';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    bool auto_mission_reload
    bool auto_engage
    bool auto_mission_change
    bool use_fms
    bool disuse_vector_map
    bool sim_mode
    bool insert_stop_line_wp
    uint32 num_of_steer_behind
    float64 change_threshold_dist
    float64 change_threshold_angle
    float64 goal_threshold_dist
    float64 goal_threshold_vel
    float64 stopped_vel
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ConfigDecisionMaker(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.auto_mission_reload !== undefined) {
      resolved.auto_mission_reload = msg.auto_mission_reload;
    }
    else {
      resolved.auto_mission_reload = false
    }

    if (msg.auto_engage !== undefined) {
      resolved.auto_engage = msg.auto_engage;
    }
    else {
      resolved.auto_engage = false
    }

    if (msg.auto_mission_change !== undefined) {
      resolved.auto_mission_change = msg.auto_mission_change;
    }
    else {
      resolved.auto_mission_change = false
    }

    if (msg.use_fms !== undefined) {
      resolved.use_fms = msg.use_fms;
    }
    else {
      resolved.use_fms = false
    }

    if (msg.disuse_vector_map !== undefined) {
      resolved.disuse_vector_map = msg.disuse_vector_map;
    }
    else {
      resolved.disuse_vector_map = false
    }

    if (msg.sim_mode !== undefined) {
      resolved.sim_mode = msg.sim_mode;
    }
    else {
      resolved.sim_mode = false
    }

    if (msg.insert_stop_line_wp !== undefined) {
      resolved.insert_stop_line_wp = msg.insert_stop_line_wp;
    }
    else {
      resolved.insert_stop_line_wp = false
    }

    if (msg.num_of_steer_behind !== undefined) {
      resolved.num_of_steer_behind = msg.num_of_steer_behind;
    }
    else {
      resolved.num_of_steer_behind = 0
    }

    if (msg.change_threshold_dist !== undefined) {
      resolved.change_threshold_dist = msg.change_threshold_dist;
    }
    else {
      resolved.change_threshold_dist = 0.0
    }

    if (msg.change_threshold_angle !== undefined) {
      resolved.change_threshold_angle = msg.change_threshold_angle;
    }
    else {
      resolved.change_threshold_angle = 0.0
    }

    if (msg.goal_threshold_dist !== undefined) {
      resolved.goal_threshold_dist = msg.goal_threshold_dist;
    }
    else {
      resolved.goal_threshold_dist = 0.0
    }

    if (msg.goal_threshold_vel !== undefined) {
      resolved.goal_threshold_vel = msg.goal_threshold_vel;
    }
    else {
      resolved.goal_threshold_vel = 0.0
    }

    if (msg.stopped_vel !== undefined) {
      resolved.stopped_vel = msg.stopped_vel;
    }
    else {
      resolved.stopped_vel = 0.0
    }

    return resolved;
    }
};

module.exports = ConfigDecisionMaker;
