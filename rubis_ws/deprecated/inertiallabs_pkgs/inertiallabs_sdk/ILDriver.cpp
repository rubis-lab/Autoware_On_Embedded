#include "ILDriver.h"
#include "UDDParser.h"
#include "SerialPort.h"
#include "NetClient.h"
#include <iostream>
#include <string>
#include <errno.h>

namespace IL {

	Driver::Driver()
		: latestData()
		, deviceInfo()
		, deviceParam()
		, port(nullptr)
		, workerThread(nullptr)
		, quit(false)
		, devInfoRead(false)
		, onRequestMode(false)
		, sessionState(Off)
		, callback(nullptr)
	{
		port = nullptr;
	}

	Driver::~Driver()
	{
		disconnect();
	}

	int Driver::connect(const char* url)
	{
		if (workerThread)
			return 1;
		std::string urlStr(url);
		std::string pathStr;
		size_t pos = urlStr.find(':');
		if (pos != std::string::npos) {
			std::string typeStr = urlStr.substr(0, pos);
			if ("serial" == typeStr) {
				port = new SerialPort;
				pathStr = urlStr.substr(pos + 1);
			}
			else if ("tcp" == typeStr || "udp" == typeStr) {
				port = new NetClient;
				pathStr = urlStr;
			}
			else
				return 256;
		}
		else {
			return 256;
		}
		int result = port->open(pathStr.c_str());
		if (result) {
			disconnect();
			return result;
		}
		sessionState = Off;
		workerThread = new std::thread(threadFunc, this);
		readDevInfo();
		if (sessionState < GotDevParams)
		{
			disconnect();
			return 512;
		}
		return 0;
	}

	void Driver::disconnect()
	{
		if (workerThread) {
			quit = true;
			workerThread->join();
			delete workerThread;
			workerThread = nullptr;
		}
		if (port) {
			if (port->isOpen()) {
				port->close();
			}
			delete port;
			port = nullptr;
		}
	}

	int Driver::start(unsigned char mode, bool onRequest, const char* logname)
	{
		char command = onRequest ? '\xC1' : mode;
		if (sessionState != GotDevParams) {
			return 1;
		}

		onRequestMode = onRequest;
		sendPacket(0, &command, 1);
		for (int repeat = 0; repeat < 10; ++repeat) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			if (sessionState >= GetIntinialReport) {
				break;
            }
		}
		if (sessionState < GetIntinialReport) {
			return 2;
		}
		if (sessionState < Processing) {
			std::this_thread::sleep_for(std::chrono::seconds(deviceParam.initAlignmentTime));
			for (int repeat = 0; repeat < 10; ++repeat) {
				if (sessionState == Processing) {
					break;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			if (sessionState < Processing) {
				return 3;
			}
		}
		if (logname)
			log.open(logname);
		return 0;
	}

	int Driver::request(unsigned char mode, int timeout)
	{
		if (!onRequestMode)
			return -1;
		if (sessionState < GetIntinialReport)
			return -2;
		requestFulfilled = false;
		requestCode = mode;
		sendPacket(0, &requestCode, 1);
		for (int i = 0; i < timeout; ++i)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			if (requestFulfilled)
				return 0;
		}
		return 1;
	}

	int Driver::stop()
	{
		sessionState = Closing;
		for (int trial = 0; trial < 5; ++trial)
		{
			sendPacket(0, "\xFE", 1);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		if (log) {
			log.close();
		}
		sessionState = Off;
		if (!devInfoRead)					// In case the device was in auto-start mode or already started when we connected
			return readDevInfo();
		sessionState = GotDevParams;
		return 0;
	}

	INSDeviceInfo Driver::getDeviceInfo()
	{
		return deviceInfo;
	}

	INSDevicePar Driver::getDeviceParams()
	{
		return deviceParam;
	}

	void Driver::setCallback(void (*newCallback)(INSDataStruct*, void*), void * userContext)
	{
		callback = newCallback;
		callbackContext = userContext;
	}

	int Driver::sendPacket(char type, const char* payload, unsigned int size)
	{
		uint8_t buf[65536] = "\xAA\x55\x00\x00";
		uint16_t checksum = buf[2] = type;
		checksum += buf[4] = (size + 6) & 0xFF;
		checksum += buf[5] = (size + 6) >> 8;
		for (unsigned int i = 0; i < size; ++i)
			checksum += buf[6 + i] = payload[i];
		buf[6 + size] = checksum & 0xFF;
		buf[7 + size] = checksum >> 8;
		return port->write(reinterpret_cast<char*>(buf), 8 + size);
	}

	int Driver::readDevInfo()
	{
		int result = sendPacket(0, "\x12", 1);(void) result;
		for (int sec = 0; sec < 30; ++sec) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			if (sessionState)
				break;
		}
		if (!sessionState)
		{
			disconnect();
			return 2;
		}
		sendPacket(0, "\x41", 1);
		for (int sec = 0; sec < 10; ++sec) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			if (GotDevParams == sessionState || Processing == sessionState)
				break;
		}
		return 0;
	}

	void Driver::readerLoop()
	{
		enum TrafficType
		{
			Invalid,
			Binary,
			RawImu,
			Nmea,
			Nmea_A,
		};
		int state = 0;
		int checksum = 0;
		char buf[65536];
		UDDParser parser;
		uint16_t len = 0;
		std::string NMEA;
		std::string RawIMU;
		uint32_t RawIMUcounter;
		uint8_t header[3];
		TrafficType trafficType = Invalid;
		uint8_t byte = 0, prevByte = 0;
		while (!quit)
		{
			int readBytes = port->read(buf, sizeof(buf));
			if (readBytes < 0)
				quit = true;
			else
			{
				for (int i = 0; i < readBytes; ++i)
				{
					prevByte = byte;
					byte = buf[i];
					if (state) checksum += byte;
					switch (state)
					{
					case 0:
						for (int i = 0; i < 2; ++i)
							header[i] = header[i + 1];
						header[2] = byte;
						if (0xAA == header[0] && 0x55 == header[1] && 0x01 == header[2])
						{
							state = 3;
							trafficType = Binary;
							checksum = 1;
						}
						else if (0xAA == header[0] && 0x44 == header[1] && 0x12 == header[2])
						{
							state = 3;
							trafficType = RawImu;
						}
						else if (0x0D == header[0] && 0x0A == header[1] && '$' == header[2])
						{
							state = 3;
							trafficType = Nmea;
						}
						else if (0x0D == header[0] && 0x0A == header[1] && 'A' == header[2])
						{
							state = 3;
							trafficType = Nmea_A;
						}
						break;
					case 3:
						switch (trafficType)
						{
						case Binary:
							parser.code = byte;
							++state;
							break;
						case RawImu:
							RawIMU = "\xAA\x44\x12";
							RawIMU += (char)byte;
							RawIMUcounter = 4;
							++state;
							break;
						case Nmea:
							NMEA = "$";
							NMEA += (char)byte;
							++state;
							break;
						case Nmea_A:
							NMEA = "A";
							NMEA += (char)byte;
							++state;
							break;
						default:
							break;
						}
						break;
					case 4:
						switch (trafficType)
						{
						case Binary:
							parser.payloadLen = byte;
							++state;
							break;
						case RawImu:
							if (72 == ++RawIMUcounter)
							{
								state = 0;
								{
									// TODO: RAWIMUB data parser
								}
							}
							break;
						case Nmea:
							if (0x0a == byte && 0x0d == prevByte)
							{
								header[1] = prevByte; header[2] = byte;
								state = 0;
								{
									// We do not parse NMEA
								}
							}
							break;
						case Nmea_A:
							if (0x0a == byte && 0x0d == prevByte)
							{
								header[1] = prevByte; header[2] = byte;
								state = 0;
								{
									// We do not parse COBHAM
								}
							}
							break;
						default:
							break;
						}
						break;
					case 5:
						parser.payloadLen += static_cast<uint16_t>(byte) << 8;
						++state;
						parser.payloadInd = 0;
						if (6 == parser.payloadLen)
							++state; 		// no payload;
						if (parser.payloadLen < 6)
							state = 0; 		// The packet length value cannot be less than 6
						parser.payloadLen -= 6;
						break;
					case 6:
						parser.payloadBuf[parser.payloadInd] = byte;
						if (++parser.payloadInd == parser.payloadLen)
							++state;
						break;
					case 7:
						checksum -= byte;
						checksum -= byte;
						if (checksum & 0xFF)
							state = 0;
						else
							++state;
						checksum >>= 8;
						break;
					case 8:
						checksum -= byte;
						checksum -= byte;
						if (!(checksum & 0xFF))
						{
							switch (parser.code) {
							case 0x12:
								if (parser.payloadLen == sizeof(INSDeviceInfo))
								{
									sessionState = GotDevInfo;
									deviceInfo = *reinterpret_cast<INSDeviceInfo*>(parser.payloadBuf);
								}
								break;
							case 0x41:
								if (parser.payloadLen == sizeof(INSDevicePar))
								{
									sessionState = GotDevParams;
									devInfoRead = true;
									deviceParam = *reinterpret_cast<INSDevicePar*>(parser.payloadBuf);
									sendPacket(0, "\xB1\x6C", 2);
								}
								break;
							case 0xB1:
								if (parser.payloadLen == 2)
								{
									if (parser.payloadBuf[0] == 0x6C)
									{
										parser.high_precision_heave = (parser.payloadBuf[1] == 0x01);
									}
								}
								break;
							default:
								if ((!sessionState || GetIntinialReport == sessionState) && (0x32 == parser.payloadLen || 0x80 == parser.payloadLen))
								{
									// Initial alignment report
									sessionState = Processing;
								}
								else if (!sessionState ||
										 GotDevParams == sessionState ||
										 GetIntinialReport == sessionState || // Ignore missing initial alignment report
										 Processing == sessionState)
								{
									if (parser.parse()) {
										sessionState = GetIntinialReport;	// ACK received
									} else {
										sessionState = Processing;
										if (log) {
											if (parser.hdrStream.str().size()) {
												log << parser.hdrStream.str() << std::endl;
											}
											log << parser.txtStream.str() << std::endl;
										}
										latestData = parser.outData;
										if (onRequestMode && parser.code == requestCode)
											requestFulfilled = true;
										if (callback) callback(&latestData, callbackContext);
									}
								}
							}
						}
					/* fall through */
					default:
						state = 0;
					}
				}
			}
		}
	}
}
