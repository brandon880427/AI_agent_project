#pragma once

// Copy this file to compile/secrets.h and fill in your real values.
// NOTE: compile/secrets.h should NOT be committed.

#define WIFI_SSID_STR "YOUR_WIFI_SSID"
#define WIFI_PASS_STR "YOUR_WIFI_PASSWORD"

// Set to your machine's LAN IP so the ESP32 can reach the FastAPI server
#define SERVER_HOST_STR "192.168.1.100"
#define SERVER_PORT_NUM 8081

// Optional UDP target (only used when ENABLE_UDP=1)
#define UDP_HOST_STR "127.0.0.1"
#define UDP_PORT_NUM 12345
