#pragma once

namespace IL {
	class Transport
	{
	public:
		Transport() {};
		virtual ~Transport() {};
		virtual int open(const char* path) { (void)path;return 1; }
		virtual bool isOpen() { return false; }
		virtual void close() {}
		virtual int read(char* buf, unsigned int size) { (void)buf; (void)size; return 0; }
		virtual int write(char* buf, unsigned int size) { (void)buf; (void)size; return 0; }
	};
}

