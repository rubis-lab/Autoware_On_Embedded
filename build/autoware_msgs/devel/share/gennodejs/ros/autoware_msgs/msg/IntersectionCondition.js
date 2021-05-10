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

class IntersectionCondition {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.intersectionID = null;
      this.intersectionDistance = null;
      this.isIntersection = null;
      this.riskyLeftTurn = null;
      this.riskyRightTurn = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('intersectionID')) {
        this.intersectionID = initObj.intersectionID
      }
      else {
        this.intersectionID = 0;
      }
      if (initObj.hasOwnProperty('intersectionDistance')) {
        this.intersectionDistance = initObj.intersectionDistance
      }
      else {
        this.intersectionDistance = 0.0;
      }
      if (initObj.hasOwnProperty('isIntersection')) {
        this.isIntersection = initObj.isIntersection
      }
      else {
        this.isIntersection = 0;
      }
      if (initObj.hasOwnProperty('riskyLeftTurn')) {
        this.riskyLeftTurn = initObj.riskyLeftTurn
      }
      else {
        this.riskyLeftTurn = 0;
      }
      if (initObj.hasOwnProperty('riskyRightTurn')) {
        this.riskyRightTurn = initObj.riskyRightTurn
      }
      else {
        this.riskyRightTurn = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type IntersectionCondition
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [intersectionID]
    bufferOffset = _serializer.int32(obj.intersectionID, buffer, bufferOffset);
    // Serialize message field [intersectionDistance]
    bufferOffset = _serializer.float32(obj.intersectionDistance, buffer, bufferOffset);
    // Serialize message field [isIntersection]
    bufferOffset = _serializer.int8(obj.isIntersection, buffer, bufferOffset);
    // Serialize message field [riskyLeftTurn]
    bufferOffset = _serializer.int8(obj.riskyLeftTurn, buffer, bufferOffset);
    // Serialize message field [riskyRightTurn]
    bufferOffset = _serializer.int8(obj.riskyRightTurn, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type IntersectionCondition
    let len;
    let data = new IntersectionCondition(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [intersectionID]
    data.intersectionID = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [intersectionDistance]
    data.intersectionDistance = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [isIntersection]
    data.isIntersection = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [riskyLeftTurn]
    data.riskyLeftTurn = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [riskyRightTurn]
    data.riskyRightTurn = _deserializer.int8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 11;
  }

  static datatype() {
    // Returns string type for a message object
    return 'autoware_msgs/IntersectionCondition';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ec2240e9bbead2818bb2ecd5340b0db8';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    int32 intersectionID
    float32 intersectionDistance
    int8 isIntersection
    int8 riskyLeftTurn
    int8 riskyRightTurn
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
    const resolved = new IntersectionCondition(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.intersectionID !== undefined) {
      resolved.intersectionID = msg.intersectionID;
    }
    else {
      resolved.intersectionID = 0
    }

    if (msg.intersectionDistance !== undefined) {
      resolved.intersectionDistance = msg.intersectionDistance;
    }
    else {
      resolved.intersectionDistance = 0.0
    }

    if (msg.isIntersection !== undefined) {
      resolved.isIntersection = msg.isIntersection;
    }
    else {
      resolved.isIntersection = 0
    }

    if (msg.riskyLeftTurn !== undefined) {
      resolved.riskyLeftTurn = msg.riskyLeftTurn;
    }
    else {
      resolved.riskyLeftTurn = 0
    }

    if (msg.riskyRightTurn !== undefined) {
      resolved.riskyRightTurn = msg.riskyRightTurn;
    }
    else {
      resolved.riskyRightTurn = 0
    }

    return resolved;
    }
};

module.exports = IntersectionCondition;
