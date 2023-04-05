#pragma once
#include <cstdint>

namespace IL {

	namespace PacketType {
		const uint8_t Cobham_UAV200_Satcom = 0x46;
		const uint8_t IL_Sensors = 0x50;
		const uint8_t IL_OPVT = 0x52;
		const uint8_t IL_min = 0x53;
		const uint8_t IL_NMEA = 0x54;
		const uint8_t IL_Sensors_NMEA = 0x55;
		const uint8_t IL_QPVT = 0x56;
		const uint8_t IL_OPVT2A = 0x57;
		const uint8_t IL_OPVT2AHR = 0x58;
		const uint8_t IL_OPVT2AW = 0x59;
		const uint8_t IL_OPVTAD = 0x61;
		const uint8_t MRU_OPVTHSSHR = 0x64;
		const uint8_t IL_OPVT_rawIMU = 0x66;
		const uint8_t IL_OPVT_GNSSext = 0x67;
		const uint8_t SPAN_rawIMU = 0x68;
		const uint8_t IL_UDD = 0x95;
	};

	struct INSDataStruct
	{
		unsigned int ms_gps;
		double GPS_INS_Time;
		double GPS_IMU_Time;
		int UTC_Hour;
		int UTC_Minute;
		int UTC_Second;
		double UTC_DecSec;
		int UTC_Day;
		int UTC_Month;
		int UTC_Year;
		uint64_t UTCSecSinceEpoch;
		double Heading;
		double Pitch;
		double Roll;
		double Quat[4];
		double Latitude;
		double Longitude;
		double Altitude;
		double VelENU[3];
		double Gyro[3];
		double Acc[3];
		double AccPVPoint[3];
		double Mag[3];
		double pBar;
		double hBar;
		double GBias[3];
		double ABias[3];
		double LatGNSS;
		double LonGNSS;
		double AltGNSS;
		double V_Hor;
		double Trk_gnd;
		double V_ver;
		double Heading_GNSS;
		double Pitch_GNSS;
		double LatGNSSStd;
		double LonGNSSStd;
		double AltGNSSStd;
		double HeadingGNSSStd;
		double PitchGNSSStd;
		int GNSSInfo1;
		int GNSSInfo2;
		int SVtrack;
		int SVsol;
		int SVsolL1;
		int SVSolMulti;
		int GalBD;
		int GPSGlo;
		int TimeStatus;
		int ExtSolStatus;
		int GNSSSolStatus;
		int GNSSSolType;
		int AnglesType;
		int Week;
		int GNSSVelLatency;
		int GNSSPosMs;
		int GNSSVelMs;
		int GNSSHdgMs;
		int NewGPS;
		double GDOP;
		double PDOP;
		double HDOP;
		double VDOP;
		double TDOP;
		double GNSS_PACC;
		double GNSS_VACC;
		double VSup;
		double VStab;
		double Temp;
		int USW;
		int INSSolStatus;
		double KFLatStd;
		double KFLonStd;
		double KFAltStd;
		double KFHdgStd;
		double KFVelStd[3];
		double Odometer;
		double AirSpeed;
		double WindN;
		double WindE;
		double WindNStd;
		double WindEStd;
		double LatExt;
		double LonExt;
		double AltExt;
		double LatExtStd;
		double LonExtStd;
		double AltExtStd;
		double ExtPosLatency;
		double LocLat;
		double LocLon;
		double LocAlt;
		double LocDopplerShift;
		double LocDopplerShiftStd;
		double ExtAntPri[3];
		double ExtAntSec[3];
		int NewAiding;
		double HdgExt;
		double HdgExtStd;
		double HdgExtLatency;
		double DVLRight;
		double DVLFwd;
		double DVLUp;
		double DVLRightStd;
		double DVLFwdStd;
		double DVLUpStd;
		double DVLLatency;
		double DVLPressure;
		double GBExt[3];
		double ABExt[3];
		double PitchExt;
		double RollExt;
		int Latency_ms_pos;
		int Latency_ms_vel;
		int Latency_ms_head;
		int UP;
		int UT;
		double GNSS_ECEF_X;
		double GNSS_ECEF_Y;
		double GNSS_ECEF_Z;
		double GNSS_ECEF_VX;
		double GNSS_ECEF_VY;
		double GNSS_ECEF_VZ;
		double GNSS_ECEF_VXStd;
		double GNSS_ECEF_VYStd;
		double GNSS_ECEF_VZStd;
		double DiffAge;
		int LatencyECEF;
		int PPPStore;
		int PPPApp;
		uint64_t dataPresent[8];

		double Heave;
		double Surge;
		double Sway;
		double Heave_velocity;
		double Surge_velocity;
		double Sway_velocity;
		double significant_wave_height;
	};


#ifdef _MSC_VER
#pragma pack(push,1)
#endif
	struct INSDeviceInfo {
		char IDN[8];
		char FW[40];
		uint8_t pressSensor;
		uint8_t imuType;
		char imuSN[8];
		char imuFW[40];
		char GNSSmodel[16];
		char GNSSsn[16];
		char GNSShw[16];
		char GNSSfw[16];
		uint16_t week;
		uint8_t GNSSmaxRate;
		uint8_t reserved;
#ifdef __GNUC__
	} __attribute__((packed));
#else
};
#endif
	struct INSDevicePar {
		uint16_t dataRate;
		uint16_t initAlignmentTime;
		int32_t magDeclination;
		int32_t Lat;
		int32_t Lon;
		int32_t Alt;
		uint8_t YearSince1900;
		uint8_t Month;
		uint8_t Day;
		int16_t AlignmentAngles[3];
		int16_t COGtoINSleverArm[3];
		int16_t AntennaToINSleverArm[3];
		uint8_t GeoidAltitude;
		char reserved1[8];
		char IDN[8];
		uint8_t enableBaroAltimeter;
		char reserved2;
#ifdef __GNUC__
	} __attribute__((packed));
#else
};
#endif
#ifdef _MSC_VER
#pragma pack(pop)
#endif

}
