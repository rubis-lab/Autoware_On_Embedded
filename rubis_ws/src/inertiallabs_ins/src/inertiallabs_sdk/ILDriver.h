#pragma once
#include <thread>
#include <fstream>
#include "dataStructures.h"
#include "Transport.h"

namespace IL {
	class Driver
	{
	public:

        enum SessionState
        {
            Off,
            GotDevInfo,
            GotDevParams,
            GetIntinialReport,
            Processing,
            Closing
        };

		Driver();
		~Driver();
		int connect(const char* url);
		void disconnect();
		int start(unsigned char mode, bool onRequest = false, const char* logname = nullptr);
		int request(unsigned char mode, int timeout);
		int stop();
		INSDeviceInfo getDeviceInfo();
		INSDevicePar getDeviceParams();
		bool isStarted() { return sessionState == Processing; }
		void setCallback(void (*newCallback)(INSDataStruct*, void*), void* userContext);

	private:
		INSDataStruct latestData;
		INSDeviceInfo deviceInfo;
		INSDevicePar deviceParam;
		Transport* port;
		std::thread* workerThread;
		bool quit;
		bool devInfoRead;
		bool onRequestMode;
		char requestCode;
		bool requestFulfilled;
		SessionState sessionState;
		int sendPacket(char type, const char* payload, unsigned int size);
		int readDevInfo();
		void readerLoop();
		static void threadFunc(Driver* instance) { instance->readerLoop(); }
		std::ofstream log;
		void (*callback)(INSDataStruct*, void*);
		void* callbackContext;
	};

}

