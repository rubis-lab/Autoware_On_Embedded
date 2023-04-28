#include "ILDriver.h"
#include <iostream>


void callback(IL::INSDataStruct* data, void* context)
{
    std::cout << *reinterpret_cast<std::string*>(context) << " Time: " << data->GPS_INS_Time << ", USW = " << data->USW << std::endl;
}

int main()
{
    IL::Driver driver;
    uint8_t packetType = IL::PacketType::IL_OPVT;
    std::cout << "Enter INS URL ([serial/tcp/udp]:[com port path / hostname or IP]:[baud rate or TCP/UDP port]: ";
    std::string URL;
    std::cin >> URL;
    if (driver.connect(URL.c_str()) != 0)
    {
        std::cout << "IL Driver failed to connect!" << std::endl;
        return 1;
    }
    if (driver.isStarted()) {
        driver.stop();
    }
    IL::INSDeviceInfo devInfo = driver.getDeviceInfo();
    IL::INSDevicePar devParams = driver.getDeviceParams();
    std::cout << "Found INS " << std::string(devInfo.IDN,8) << " providing data at " << devParams.dataRate << " Hz" << std::endl;
    std::string context(devInfo.IDN, 8);
    int res;
    driver.setCallback(&callback, &context);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Starting device in polling mode" << std::endl;
    if ((res = driver.start(packetType, true)))
    {
        std::cout << "INS failed to start! " << res << std::endl;
        driver.disconnect();
        return 2;
    }
    for (int i = 0; i < 5; ++i)
    {
        if (driver.request(0x52, 100) == 0) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else
        {
            std::cout << "INS failed to fulfill request!" << std::endl;
            break;
        }
    }
    std::cout << "Stopping device" << std::endl;
    driver.stop();
    std::cout << "Starting device in continuous mode" << std::endl;
    if ((res = driver.start(packetType, false)))
    {
        std::cout << "INS failed to start! " << res << std::endl;
        driver.disconnect();
        return 2;
    }
    for (int i = 0; i < 5; ++i)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "Stopping device" << std::endl;
    driver.stop();
    std::cout << "Closing device" << std::endl;
    driver.disconnect();
    return 0;
}
