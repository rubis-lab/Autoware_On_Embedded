// Auto-generated. Do not edit!

// (in-package autoware_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let RUBISTrafficSignal = require('./RUBISTrafficSignal.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class RUBISTrafficSignalArray {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.signals = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('signals')) {
        this.signals = initObj.signals
      }
      else {
        this.signals = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RUBISTrafficSignalArray
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [signals]
    // Serialize the length for message field [signals]
    bufferOffset = _serializer.uint32(obj.signals.length, buffer, bufferOffset);
    obj.signals.forEach((val) => {
      bufferOffset = RUBISTrafficSignal.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RUBISTrafficSignalArray
    let len;
    let data = new RUBISTrafficSignalArray(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [signals]
    // Deserialize array length for message field [signals]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.signals = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.signals[i] = RUBISTrafficSignal.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += 12 * object.signals.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'autoware_msgs/RUBISTrafficSignalArray';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '8ae2769f49d9241a71af08054e4cc568';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    RUBISTrafficSignal[] signals
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
    
    ================================================================================
    MSG: autoware_msgs/RUBISTrafficSignal
    int32 id
    int32 type
    float32 time
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RUBISTrafficSignalArray(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.signals !== undefined) {
      resolved.signals = new Array(msg.signals.length);
      for (let i = 0; i < resolved.signals.length; ++i) {
        resolved.signals[i] = RUBISTrafficSignal.Resolve(msg.signals[i]);
      }
    }
    else {
      resolved.signals = []
    }

    return resolved;
    }
};

module.exports = RUBISTrafficSignalArray;
