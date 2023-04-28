#pragma once
#include "Transport.h"

namespace IL {

	class SerialPort : public Transport
	{
	public:
		SerialPort();
		virtual ~SerialPort();
		virtual int open(const char* url);
		virtual bool isOpen();
		virtual void close();
		virtual int read(char* buf, unsigned int size);
		virtual int write(char* buf, unsigned int size);

	private:
		int fd;
		void* hCom;
		int timeout;
	};
}

