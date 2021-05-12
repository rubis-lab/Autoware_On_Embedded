// Auto-generated. Do not edit!

// (in-package autoware_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class State {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.vehicle_state = null;
      this.mission_state = null;
      this.behavior_state = null;
      this.motion_state = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('vehicle_state')) {
        this.vehicle_state = initObj.vehicle_state
      }
      else {
        this.vehicle_state = '';
      }
      if (initObj.hasOwnProperty('mission_state')) {
        this.mission_state = initObj.mission_state
      }
      else {
        this.mission_state = '';
      }
      if (initObj.hasOwnProperty('behavior_state')) {
        this.behavior_state = initObj.behavior_state
      }
      else {
        this.behavior_state = '';
      }
      if (initObj.hasOwnProperty('motion_state')) {
        this.motion_state = initObj.motion_state
      }
      else {
        this.motion_state = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type State
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [vehicle_state]
    bufferOffset = _serializer.string(obj.vehicle_state, buffer, bufferOffset);
    // Serialize message field [mission_state]
    bufferOffset = _serializer.string(obj.mission_state, buffer, bufferOffset);
    // Serialize message field [behavior_state]
    bufferOffset = _serializer.string(obj.behavior_state, buffer, bufferOffset);
    // Serialize message field [motion_state]
    bufferOffset = _serializer.string(obj.motion_state, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type State
    let len;
    let data = new State(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [vehicle_state]
    data.vehicle_state = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [mission_state]
    data.mission_state = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [behavior_state]
    data.behavior_state = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [motion_state]
    data.motion_state = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += object.vehicle_state.length;
    length += object.mission_state.length;
    length += object.behavior_state.length;
    length += object.motion_state.length;
    return length + 16;
  }

  static datatype() {
    // Returns string type for a message object
    return 'autoware_msgs/State';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '937d7a6d5eba6aee9608f59553aebbe8';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    string vehicle_state
    string mission_state
    string behavior_state
    string motion_state
    
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
    const resolved = new State(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.vehicle_state !== undefined) {
      resolved.vehicle_state = msg.vehicle_state;
    }
    else {
      resolved.vehicle_state = ''
    }

    if (msg.mission_state !== undefined) {
      resolved.mission_state = msg.mission_state;
    }
    else {
      resolved.mission_state = ''
    }

    if (msg.behavior_state !== undefined) {
      resolved.behavior_state = msg.behavior_state;
    }
    else {
      resolved.behavior_state = ''
    }

    if (msg.motion_state !== undefined) {
      resolved.motion_state = msg.motion_state;
    }
    else {
      resolved.motion_state = ''
    }

    return resolved;
    }
};

module.exports = State;
