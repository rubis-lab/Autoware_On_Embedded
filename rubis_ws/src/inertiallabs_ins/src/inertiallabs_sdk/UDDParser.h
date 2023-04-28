#pragma once
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string.h>
#include "dataStructures.h"


namespace IL {

	class UDDParser
	{
	public:
		UDDParser();
		~UDDParser();

		int parse();
		void writeHeader();
		void writeTxtAndData();
		void finish();
		int KA;
		double SA;
		int KG;
		double SG;
		double SO;
		double SV;
		uint8_t code;
		uint16_t payloadLen;
		int payloadInd;
		uint8_t payloadBuf[65530];
		std::string dataSet;
		std::string oldDataSet;
		std::string fileName;
		std::stringstream txtStream;
		std::stringstream hdrStream;
		std::stringstream statusStream;
		double dVal;
		float fVal;
		int64_t i64val;
		uint64_t ui64val;
		int32_t i32val;
		uint32_t ui32val;
		int16_t i16val;
		uint16_t ui16val;
		int8_t i8val;
		uint8_t ui8val;
		bool parseError;
		bool high_precision_heave;
		INSDataStruct outData;

		template<typename T>
		void readVal(T& val) {
			parseError = (payloadLen - payloadInd < sizeof(val));
			if (!parseError)
			{
				memcpy(&val, &payloadBuf[payloadInd], sizeof(val));
				payloadInd += sizeof(val);
			}
		}

		template<typename T>
		double readScaled(double scale, bool tab = false, int base = 10)
		{
			T val;
			parseError = (payloadLen - payloadInd < sizeof(val));
			if (!parseError)
			{
				memcpy(&val, &payloadBuf[payloadInd], sizeof(val));
				//			val = *reinterpret_cast<T*>(&payloadBuf[payloadInd]);
				payloadInd += sizeof(val);
			}
			double res = static_cast<double>(val) / scale;
			int precision = static_cast<int>(std::log10(scale) + 0.5);
			if (precision < 0) precision = 0;
			if (8 != base && 16 != base)
				base = 10;
			if (10 == base)
				txtStream << std::fixed << std::setprecision(precision) << res;
			else
				txtStream << std::setprecision(0) << std::setbase(base) << std::setw(sizeof(T) * 8 / std::log2(base)) << std::setfill('0') << static_cast<int>(res);
			if (tab) txtStream << "\t";
			return res;
		}
	};

}
