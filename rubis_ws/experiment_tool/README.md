# Autoware Analyzer
Collect center offset data, calculate E2E(End-to-End) respnose time of instance chain and support plotting.

## How to use
1. Launch Autoware.
2. Execute to get pose difference between vector map and current pose.
    ```
    python3 autoware_analyzer_manager.py -c <filename>
    ```
3. Stop Autoware and autoware_analyzer_manager.
4. Execute to calculate E2E response time.
    ```
    python3 autoware_analyzer_manager.py -e
    ```
5. Plot the center offset and E2E response time.
    ```
    python3 autoware_analyzer_manager.py -p
    ```