// ===== all_in_one_merged.ino — XIAO ESP32S3 Sense: Camera + Mic (PDM) + IMU (ICM42688 SPI) =====
// ===== 版本: v2.4-SPIIMU - ICM42688 改为 SPI，避开 I2S 干扰；WAV chunked 播放保持 =====

#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_camera.h>
#include <esp_system.h>
#include <ArduinoWebsockets.h>
#include "ESP_I2S.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
struct WavFmt;
#include <cstring>      // memcmp
#include <WiFiUdp.h>
#include <WiFiClient.h> 
#include <SPI.h>        // <<< 改成 SPI
#include <esp_heap_caps.h>
using namespace websockets;

// Some ESP32-S3 board configs route `Serial` to USB CDC, which may not show up
// on the same /dev/cu.* device that prints ROM boot logs. Use ets_printf for
// a reliable UART0-level log channel.
extern "C" int ets_printf(const char* fmt, ...);

// ====================================================================
// Feature flags (image-only mode)
// ====================================================================
// The user only needs camera streaming. Disable everything else to reduce
// CPU/RAM usage and eliminate unrelated tasks that can destabilize WiFi/WS.
#ifndef ENABLE_AUDIO
#define ENABLE_AUDIO 0
#endif
#ifndef ENABLE_IMU
#define ENABLE_IMU 0
#endif
#ifndef ENABLE_TTS
#define ENABLE_TTS 0
#endif
#ifndef ENABLE_UDP
#define ENABLE_UDP 0
#endif

// Guard ArduinoWebsockets usage across multiple FreeRTOS tasks.
// NOTE: The ArduinoWebsockets client is not thread-safe; without this,
// concurrent poll()/sendBinary()/connect() can cause random disconnects/resets.
struct MutexLock {
  SemaphoreHandle_t m = nullptr;
  bool locked = false;
  MutexLock(SemaphoreHandle_t m_, TickType_t waitTicks) : m(m_) {
    if (m) locked = (xSemaphoreTake(m, waitTicks) == pdTRUE);
  }
  ~MutexLock() {
    if (locked && m) xSemaphoreGive(m);
  }
};

// ===== WiFi / Server =====
#include "secrets.h"

const char* WIFI_SSID   = WIFI_SSID_STR;
const char* WIFI_PASS   = WIFI_PASS_STR;
// Set to your machine's LAN IP so the ESP32 can reach the FastAPI server
const char* SERVER_HOST = SERVER_HOST_STR;
const uint16_t SERVER_PORT = SERVER_PORT_NUM;

static const char* CAM_WS_PATH = "/ws/camera";
static const char* AUD_WS_PATH = "/ws_audio";

// ===== Camera config =====
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

// Use a smaller frame size and fewer frame buffers to avoid PSRAM/internal RAM exhaustion
framesize_t g_frame_size = FRAMESIZE_QVGA; // lower resolution to reduce RAM usage
#define JPEG_QUALITY  18
// NOTE: fb_count must be >= the number of in-flight frames.
// This firmware uses separate capture/send tasks and a queue, so fb_count==1 can
// lead to buffer reuse while an older pointer is still queued, corrupting JPEGs.
// We pick a safe default at runtime based on PSRAM availability.
static int g_cam_fb_count = 0;
volatile int g_target_fps = 0; // 新增：0=不限，>0 则按该FPS限速发送

// 【新增】视频传输性能监控
volatile unsigned long frame_captured_count = 0;  // 采集帧计数
volatile unsigned long frame_sent_count = 0;      // 发送帧计数
volatile unsigned long frame_dropped_count = 0;   // 丢弃帧计数
volatile unsigned long frame_capture_fail_total = 0; // 采集/拷贝失败累计
volatile unsigned long last_stats_time = 0;       // 上次统计时间
volatile unsigned long ws_send_fail_count = 0;    // WebSocket发送失败计数

// ===== Stream stability knobs =====
// Server closes the socket if it sees no WS messages for ~30s.
// Keep sending a tiny text keepalive so the connection stays healthy even
// during brief camera capture gaps.
static const unsigned long CAM_WS_KEEPALIVE_MS = 8000;     // text keepalive interval
static const unsigned long CAM_WS_FORCE_RECONNECT_MS = 20000; // if no send for this long, reconnect

// ===== Mic (PDM RX) =====
#define I2S_MIC_CLOCK_PIN 42
#define I2S_MIC_DATA_PIN  41
const int SAMPLE_RATE     = 16000; 
const int CHUNK_MS        = 20;
const int BYTES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS / 1000 * 2;
const int AUDIO_QUEUE_DEPTH = 10;

// ===== Speaker (I2S TX → MAX98357A) =====
#define I2S_SPK_BCLK 7
#define I2S_SPK_LRCK 8
#define I2S_SPK_DIN  9
const int TTS_RATE = 16000;

// ===== IMU (ICM42688 over SPI) / UDP =====
// 使用 D0~D3 作为 SPI
#define IMU_SPI_SCK   1   // D0
#define IMU_SPI_MOSI  2   // D1
#define IMU_SPI_MISO  3   // D2
#define IMU_SPI_CS    4   // D3
const char* UDP_HOST  = UDP_HOST_STR;
const int   UDP_PORT  = UDP_PORT_NUM;

WiFiUDP udp;

// ===== WS / Queues / I2S =====
WebsocketsClient wsCam;
// Keep wsAud symbol available for builds that enable audio, but in image-only mode
// all audio logic is compiled out via ENABLE_AUDIO.
WebsocketsClient wsAud;
// Camera WebSocket must be owned by a single task (ArduinoWebsockets is not thread-safe).
// Do not call wsCam.* from multiple tasks.
SemaphoreHandle_t wsAudMutex = nullptr;
volatile bool cam_ws_ready = false;
volatile bool aud_ws_ready = false;
volatile bool snapshot_in_progress = false; // 抓拍期间暂停实时采集
// 用于限制连接尝试频率，避免在网络不稳时快速反复创建/关闭 socket
unsigned long lastWsCamAttempt = 0;
unsigned long lastWsAudAttempt = 0;
// 新增：相機 WS 重連退避參數
unsigned long wsCam_backoff_ms = 1000; // 初始退避 1s
const unsigned long wsCam_backoff_max = 30000; // 最大退避 30s

// Debug snapshot for heartbeat without touching wsCam from loop().
volatile int g_wsCamAvailSnap = 0;

typedef struct {
  size_t len;
  uint8_t* buf;
} JpegFrame;
typedef JpegFrame* frame_ptr_t;
QueueHandle_t qFrames;

typedef struct {
  size_t n;
  uint8_t data[BYTES_PER_CHUNK];
} AudioChunk;
QueueHandle_t qAudio;

#define TTS_QUEUE_DEPTH 48
typedef struct { uint16_t n; uint8_t data[2048]; } TTSChunk;
QueueHandle_t qTTS;
volatile bool tts_playing = false;

I2SClass i2sIn;   // PDM RX (Mic)
I2SClass i2sOut;  // STD TX (Speaker)
volatile bool run_audio_stream = false;

// ====================================================================
// Camera
// ====================================================================
bool apply_framesize(framesize_t fs) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return false;
  int r = s->set_framesize(s, fs);
  if (r == 0) { g_frame_size = fs; return true; }
  return false;
}

bool init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;

  // XCLK too low can lead to unstable / corrupt JPEG streams on some sensors/boards.
  // 20MHz is the common stable default for OV2640-class modules.
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = g_frame_size;
  config.jpeg_quality = JPEG_QUALITY;
  // With only 1 framebuffer, CAMERA_GRAB_LATEST can starve esp_camera_fb_get().
  // Use at least 2 buffers even without PSRAM (QVGA JPEG fits DRAM), and more
  // when PSRAM is available.
  g_cam_fb_count      = psramFound() ? 3 : 2;
  config.fb_count     = g_cam_fb_count;
  // Prefer PSRAM if available to reduce DRAM pressure and improve stability.
  // Falls back to DRAM for variants without PSRAM.
  config.fb_location  = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.grab_mode    = (config.fb_count > 1) ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) { Serial.printf("[CAM] init failed: 0x%x\n", err); return false; }

  sensor_t * s = esp_camera_sensor_get();
  if (s) {

    s->set_hmirror(s, 1);  // ★ 新增：水平镜像，与人眼左右一致（1=开，0=关）
    s->set_vflip(s, 0);    // ★ 新增：垂直翻转；若镜头“倒装”，改为 1

    s->set_brightness(s, 0);
    s->set_contrast(s, 1);
    s->set_saturation(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_exposure_ctrl(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_aec2(s, 0);
    s->set_aec_value(s, 40);
  }
  return true;
}

static inline bool jpeg_has_marker(const uint8_t* p, size_t n, uint8_t marker) {
  if (!p || n < 4) return false;
  // Scan for 0xFF <marker>. This is safe for our small frames.
  for (size_t i = 0; i + 1 < n; ++i) {
    if (p[i] == 0xFF && p[i + 1] == marker) return true;
  }
  return false;
}

static inline bool jpeg_seems_decodable(const uint8_t* jpg, size_t jpg_len) {
  if (!jpg || jpg_len < 4) return false;
  if (!(jpg[0] == 0xFF && jpg[1] == 0xD8)) return false;               // SOI
  if (!(jpg[jpg_len - 2] == 0xFF && jpg[jpg_len - 1] == 0xD9)) return false; // EOI
  // Require Start Of Frame (baseline/progressive) + Start Of Scan.
  bool has_sof = jpeg_has_marker(jpg, jpg_len, 0xC0) || jpeg_has_marker(jpg, jpg_len, 0xC2);
  bool has_sos = jpeg_has_marker(jpg, jpg_len, 0xDA);
  // Require quantization + Huffman tables; missing tables often decode-fail.
  bool has_dqt = jpeg_has_marker(jpg, jpg_len, 0xDB);
  bool has_dht = jpeg_has_marker(jpg, jpg_len, 0xC4);
  return has_sof && has_sos && has_dqt && has_dht;
}

inline void free_frame(frame_ptr_t f) {
  if (!f) return;
  if (f->buf) {
    free(f->buf);
    f->buf = nullptr;
  }
  free(f);
}

inline void enqueue_frame(frame_ptr_t f) {
  if (!f) return;
  if (xQueueSend(qFrames, &f, 0) != pdPASS) {
    // 队列满，丢弃最旧的帧
    frame_ptr_t drop = nullptr;
    if (xQueueReceive(qFrames, &drop, 0) == pdPASS) {
      free_frame(drop);
      frame_dropped_count++;  // 统计丢帧
    }
    // 再试一次入队；如果仍失败就释放本帧
    if (xQueueSend(qFrames, &f, 0) != pdPASS) {
      free_frame(f);
      frame_dropped_count++;
    }
  }
}

void taskCamCapture(void*) {
  unsigned long last_log = 0;
  unsigned long capture_fail_count = 0;
  unsigned long consecutive_fail = 0;
  unsigned long last_reinit_ms = 0;
  const unsigned long CAM_REINIT_COOLDOWN_MS = 5000;
  const unsigned long CAM_REINIT_FAIL_THRESHOLD = 250; // consecutive failures before reinit
  
  for(;;){
    if (snapshot_in_progress) { vTaskDelay(pdMS_TO_TICKS(5)); continue; }
    
    if (cam_ws_ready) {
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        frame_captured_count++;
        if (fb->format != PIXFORMAT_JPEG || fb->len == 0 || fb->buf == nullptr) {
          esp_camera_fb_return(fb);
          capture_fail_count++;
          frame_capture_fail_total++;
        } else {
          // Copy the JPEG payload so the camera buffer can be returned immediately.
          frame_ptr_t f = (frame_ptr_t)malloc(sizeof(JpegFrame));
          if (!f) {
            consecutive_fail = 0;
            esp_camera_fb_return(fb);
            capture_fail_count++;
            frame_capture_fail_total++;
          } else {
            f->len = fb->len;
            // Prefer PSRAM for larger buffers when available.
            uint32_t caps = psramFound() ? (MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) : MALLOC_CAP_8BIT;
            f->buf = (uint8_t*)heap_caps_malloc(f->len, caps);
            if (!f->buf) {
              free(f);
              esp_camera_fb_return(fb);
              capture_fail_count++;
              frame_capture_fail_total++;
            } else {
              memcpy(f->buf, fb->buf, f->len);
              esp_camera_fb_return(fb);
              enqueue_frame(f);
            }
          }
        }
      } else {
        capture_fail_count++;
        frame_capture_fail_total++;
        consecutive_fail++;

        // If camera starts failing persistently, try a soft re-init.
        unsigned long now_ms = millis();
        if (consecutive_fail >= CAM_REINIT_FAIL_THRESHOLD && (now_ms - last_reinit_ms) > CAM_REINIT_COOLDOWN_MS) {
          last_reinit_ms = now_ms;
          Serial.printf("[CAM] too many capture fails (%lu). Reinitializing camera...\n", consecutive_fail);
          ets_printf("[CAM] too many capture fails (%lu). Reinitializing camera...\n", (unsigned long)consecutive_fail);
          // Best-effort restart.
          esp_camera_deinit();
          vTaskDelay(pdMS_TO_TICKS(200));
          if (!init_camera()) {
            Serial.println("[CAM] reinit failed");
            ets_printf("[CAM] reinit failed\n");
          } else {
            Serial.println("[CAM] reinit OK");
            ets_printf("[CAM] reinit OK\n");
          }
          consecutive_fail = 0;
        }
        vTaskDelay(pdMS_TO_TICKS(2));
      }
      
      // 每5秒打印一次采集统计
      unsigned long now = millis();
      if (now - last_log > 5000) {
        int queue_waiting = uxQueueMessagesWaiting(qFrames);
        Serial.printf("[CAM-CAP] captured=%lu, queue=%d, fail=%lu\n", 
                      frame_captured_count, queue_waiting, capture_fail_count);
        ets_printf("[CAM-CAP] captured=%lu queue=%d fail=%lu\n",
                   (unsigned long)frame_captured_count,
                   (int)queue_waiting,
                   (unsigned long)capture_fail_count);
        last_log = now;
        capture_fail_count = 0;  // 重置失败计数
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(20));
    }
  }
}

// Single-owner camera WebSocket loop: handles connect/poll/send.
void taskWsCamLoop(void*) {
  static TickType_t lastTick = 0;
  unsigned long last_log = 0;
  unsigned long last_sent_time = 0;
  unsigned long last_keepalive = 0;

  for (;;) {
    // If WiFi is down, ensure socket closed and drop frames quickly.
    if (WiFi.status() != WL_CONNECTED) {
      if (wsCam.available()) {
        wsCam.close();
      }
      cam_ws_ready = false;
      g_wsCamAvailSnap = 0;

      // Drain any queued frames to avoid memory growth while offline.
      frame_ptr_t f = nullptr;
      while (xQueueReceive(qFrames, &f, 0) == pdPASS) {
        free_frame(f);
        frame_dropped_count++;
      }
      vTaskDelay(pdMS_TO_TICKS(200));
      continue;
    }

    // Connect with backoff.
    if (!wsCam.available()) {
      cam_ws_ready = false;
      g_wsCamAvailSnap = 0;

      unsigned long now = millis();
      if (now - lastWsCamAttempt >= wsCam_backoff_ms) {
        lastWsCamAttempt = now;
        Serial.printf("[WS-CAM] attempting connect... (backoff=%lu ms)\n", wsCam_backoff_ms);
        if (wsCam.connect(SERVER_HOST, SERVER_PORT, CAM_WS_PATH)) {
          Serial.println("[WS-CAM] connected");
          cam_ws_ready = true;
          wsCam_backoff_ms = 1000;
          // Reset stats on successful connect.
          frame_sent_count = 0;
          frame_dropped_count = 0;
          ws_send_fail_count = 0;
          last_sent_time = millis();
        } else {
          Serial.printf("[WS-CAM] connect failed, freeHeap=%u\n", ESP.getFreeHeap());
          if (wsCam.available()) wsCam.close();
          wsCam_backoff_ms = min(wsCam_backoff_ms * 2, wsCam_backoff_max);
        }
      }
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    // Connected: keep pumping poll.
    wsCam.poll();
    cam_ws_ready = wsCam.available();
    g_wsCamAvailSnap = cam_ws_ready ? 1 : 0;
    if (!cam_ws_ready) {
      // poll detected close; backoff for reconnect
      lastWsCamAttempt = millis();
      wsCam_backoff_ms = min(wsCam_backoff_ms * 2, wsCam_backoff_max);
      vTaskDelay(pdMS_TO_TICKS(20));
      continue;
    }

    frame_ptr_t f = nullptr;
    bool sent_any = false;
    if (xQueueReceive(qFrames, &f, pdMS_TO_TICKS(20)) == pdPASS && f) {
      // FPS throttle
      if (g_target_fps > 0) {
        const int period_ms = 1000 / g_target_fps;
        TickType_t nowTick = xTaskGetTickCount();
        int elapsed = (nowTick - lastTick) * portTICK_PERIOD_MS;
        if (elapsed < period_ms) vTaskDelay(pdMS_TO_TICKS(period_ms - elapsed));
        lastTick = xTaskGetTickCount();
      }

      // JPEG sanity check
      const uint8_t* jpg = f->buf;
      size_t jpg_len = f->len;
      if (jpg_len < 4) {
        frame_dropped_count++;
        free_frame(f);
        continue;
      }
      if (!(jpg[0] == 0xFF && jpg[1] == 0xD8)) {
        size_t limit = min((size_t)256, jpg_len - 1);
        for (size_t i = 0; i + 1 < limit; ++i) {
          if (jpg[i] == 0xFF && jpg[i + 1] == 0xD8) {
            jpg += i;
            jpg_len -= i;
            break;
          }
        }
      }
      if (jpg_len < 2 || !(jpg[0] == 0xFF && jpg[1] == 0xD8)) {
        frame_dropped_count++;
        free_frame(f);
        continue;
      }

      // Some camera drivers append padding after EOI. If the buffer doesn't end
      // with 0xFFD9, search for the last EOI and truncate.
      if (!(jpg[jpg_len - 2] == 0xFF && jpg[jpg_len - 1] == 0xD9)) {
        int last_eoi = -1;
        // Search backwards (fast path) within a reasonable tail window.
        size_t start = (jpg_len > 2048) ? (jpg_len - 2048) : 0;
        for (size_t i = jpg_len - 2; i + 1 > start; --i) {
          if (jpg[i] == 0xFF && jpg[i + 1] == 0xD9) { last_eoi = (int)i; break; }
        }
        if (last_eoi >= 0) {
          jpg_len = (size_t)last_eoi + 2;
        } else {
          frame_dropped_count++;
          free_frame(f);
          continue;
        }
      }

      // More strict check: must contain SOF + SOS, otherwise decoders will fail.
      if (!jpeg_seems_decodable(jpg, jpg_len)) {
        frame_dropped_count++;
        static unsigned long bad_log = 0;
        unsigned long now = millis();
        if (now - bad_log > 3000) {
          Serial.printf("[CAM] drop invalid JPEG (missing SOF/SOS?) len=%u\n", (unsigned)jpg_len);
          bad_log = now;
        }
        free_frame(f);
        continue;
      }

      unsigned long send_start = millis();
      bool ok = wsCam.sendBinary((const char*)jpg, jpg_len);
      unsigned long send_time = millis() - send_start;
      if (ok) {
        frame_sent_count++;
        last_sent_time = millis();
        sent_any = true;
        if (send_time > 100) {
          Serial.printf("[CAM-SEND] WARNING: send took %lu ms (size=%u)\n", send_time, (unsigned)jpg_len);
        }
      } else {
        ws_send_fail_count++;
        Serial.println("[CAM-SEND] ERROR: sendBinary failed, closing...");
        if (wsCam.available()) wsCam.close();
        cam_ws_ready = false;
        g_wsCamAvailSnap = 0;
        lastWsCamAttempt = millis();
        wsCam_backoff_ms = min(wsCam_backoff_ms * 2, wsCam_backoff_max);
      }

      free_frame(f);
    }

    // Keepalive: send a tiny text message periodically so server won't
    // consider the connection stalled during brief camera hiccups.
    unsigned long now_ms = millis();
    if (cam_ws_ready && (now_ms - last_keepalive) >= CAM_WS_KEEPALIVE_MS) {
      last_keepalive = now_ms;
      bool ok = wsCam.send("KA");
      if (ok) {
        last_sent_time = now_ms;
        sent_any = true;
      } else {
        ws_send_fail_count++;
        Serial.println("[WS-CAM] keepalive send failed, closing...");
        if (wsCam.available()) wsCam.close();
        cam_ws_ready = false;
        g_wsCamAvailSnap = 0;
        lastWsCamAttempt = millis();
        wsCam_backoff_ms = min(wsCam_backoff_ms * 2, wsCam_backoff_max);
      }
    }

    // Watchdog: if we haven't been able to send anything for too long,
    // force reconnect to break half-open socket states.
    if (cam_ws_ready) {
      unsigned long gap = (last_sent_time > 0) ? (now_ms - last_sent_time) : 0;
      if (gap >= CAM_WS_FORCE_RECONNECT_MS) {
        Serial.printf("[WS-CAM] no sends for %lu ms, forcing reconnect...\n", gap);
        ets_printf("[WS-CAM] no sends for %lu ms, forcing reconnect...\n", (unsigned long)gap);
        if (wsCam.available()) wsCam.close();
        cam_ws_ready = false;
        g_wsCamAvailSnap = 0;
        lastWsCamAttempt = millis();
        wsCam_backoff_ms = min(wsCam_backoff_ms * 2, wsCam_backoff_max);
      }
    }

    // stats log
    if (now_ms - last_log > 5000) {
      unsigned long gap = (last_sent_time > 0) ? (now_ms - last_sent_time) : 0;
      Serial.printf("[CAM] sent=%lu, dropped=%lu, ws_fail=%lu, q=%u, last_gap=%lu ms\n",
                    frame_sent_count, frame_dropped_count, ws_send_fail_count,
                    (unsigned)uxQueueMessagesWaiting(qFrames), gap);
      last_log = now_ms;
    }

    vTaskDelay(pdMS_TO_TICKS(2));
  }
}
// ====================================================================
// Mic (PDM RX)
// ====================================================================
void init_i2s_in(){
  i2sIn.setPinsPdmRx(I2S_MIC_CLOCK_PIN, I2S_MIC_DATA_PIN);
  if (!i2sIn.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("[I2S IN] init failed");
    while(1) { delay(1000); }
  }
  Serial.println("[I2S IN] PDM RX @16kHz 16bit MONO ready");
}

void taskMicCapture(void*){
  const int samples_per_chunk = BYTES_PER_CHUNK / 2; // int16
  for(;;){
    if (run_audio_stream && aud_ws_ready) {
      AudioChunk ch; ch.n = BYTES_PER_CHUNK;
      int16_t* out = reinterpret_cast<int16_t*>(ch.data);
      int i = 0;
      while (i < samples_per_chunk){
        int v = i2sIn.read();
        if (v == -1) { delay(1); continue; }
        out[i++] = (int16_t)v;
      }
      if (xQueueSend(qAudio, &ch, 0) != pdPASS){
        AudioChunk dump;
        xQueueReceive(qAudio, &dump, 0);
        xQueueSend(qAudio, &ch, 0);
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(5));
    }
  }
}

void taskMicUpload(void*){
  for(;;){
    // 只有在音频 WS 可用且标记为就绪时才发送，避免对无效 socket 的调用
    if (run_audio_stream && aud_ws_ready){
      AudioChunk ch;
      if (xQueueReceive(qAudio, &ch, pdMS_TO_TICKS(100)) == pdPASS){
        bool ok = false;
        bool send_lock_acquired = false;
        {
          MutexLock lock(wsAudMutex, pdMS_TO_TICKS(50));
          if (lock.locked) {
            send_lock_acquired = true;
            if (wsAud.available()) {
              ok = wsAud.sendBinary((const char*)ch.data, ch.n);
            }
          }
        }

        // If mutex is busy, drop this chunk (backpressure) but don't kill the WS.
        if (!send_lock_acquired) {
          continue;
        }
        if (!ok) {
          Serial.println("[AUD-UPL] ERROR: sendBinary failed, closing audio WS");
          {
            MutexLock lock(wsAudMutex, pdMS_TO_TICKS(100));
            if (lock.locked && wsAud.available()) wsAud.close();
          }
          aud_ws_ready = false;
        }
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }
}

// ====================================================================
// Speaker (I2S TX) + HTTP /stream.wav (chunked-safe)
// ====================================================================
void init_i2s_out(){
  i2sOut.setPins(I2S_SPK_BCLK, I2S_SPK_LRCK, I2S_SPK_DIN);
  if (!i2sOut.begin(I2S_MODE_STD, TTS_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("[I2S OUT] init failed");
    while(1){ delay(1000); }
  }
  Serial.println("[I2S OUT] STD TX @16kHz 32bit STEREO ready");
}

struct WavFmt {
  uint16_t audioFormat;   // 1=PCM
  uint16_t numChannels;   // 1=mono
  uint32_t sampleRate;    // 16000
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample; // 16
};

static inline void mono16_to_stereo32_msb(const int16_t* in, size_t nSamp, int32_t* outLR, float gain = 0.7f) {
  for (size_t i = 0; i < nSamp; ++i) {
    int32_t s = (int32_t)((float)in[i] * gain);
    int32_t v32 = s << 16;
    outLR[i*2 + 0] = v32;
    outLR[i*2 + 1] = v32;
  }
}

// === chunked 读取辅助 ===
static bool read_line(WiFiClient& cli, String& line, uint32_t timeout_ms=3000){
  line = "";
  uint32_t t0 = millis();
  while (millis() - t0 < timeout_ms){
    while (cli.available()){
      char ch = (char)cli.read();
      if (ch == '\n'){
        if (line.endsWith("\r")) line.remove(line.length()-1);
        return true;
      }
      line += ch;
    }
    delay(1);
  }
  return false;
}

static bool readN_http_body(WiFiClient& cli, uint8_t* buf, size_t n, bool chunked, size_t& chunk_left, uint32_t timeout_ms=3000){
  size_t got = 0;
  uint32_t t0 = millis();

  while (got < n){
    if (!cli.connected()) return false;
    if (!chunked){
      int avail = cli.available();
      if (avail > 0){
        int toread = (int)min((size_t)avail, n - got);
        int r = cli.read(buf + got, toread);
        if (r > 0) got += r;
      } else {
        if (millis() - t0 > timeout_ms) return false;
        delay(1);
      }
    } else {
      if (chunk_left == 0){
        String szline;
        if (!read_line(cli, szline, timeout_ms)) return false;
        int sc = szline.indexOf(';');
        if (sc >= 0) szline = szline.substring(0, sc);
        szline.trim();
        unsigned long sz = strtoul(szline.c_str(), nullptr, 16);
        if (sz == 0){
          String dummy;
          read_line(cli, dummy, 500);
          return false;
        }
        chunk_left = (size_t)sz;
      }
      int avail = cli.available();
      if (avail > 0){
        size_t want = min(n - got, chunk_left);
        int toread = (int)min((size_t)avail, want);
        int r = cli.read(buf + got, toread);
        if (r > 0){
          got += r;
          chunk_left -= (size_t)r;
          if (chunk_left == 0){
            while (cli.available() < 2) { if (millis() - t0 > timeout_ms) return false; delay(1); }
            cli.read(); cli.read();
          }
        }
      } else {
        if (millis() - t0 > timeout_ms) return false;
        delay(1);
      }
    }
  }
  return true;
}

static bool parse_wav_header(WiFiClient& cli, WavFmt& fmt, uint32_t& dataRemaining, bool chunked, size_t& chunk_left){
  uint8_t hdr12[12];
  if (!readN_http_body(cli, hdr12, 12, chunked, chunk_left)) return false;
  if (memcmp(hdr12, "RIFF", 4) != 0 || memcmp(hdr12 + 8, "WAVE", 4) != 0) return false;

  bool gotFmt = false;
  dataRemaining = 0;

  while (true) {
    uint8_t chdr[8];
    if (!readN_http_body(cli, chdr, 8, chunked, chunk_left)) return false;
    uint32_t sz = (uint32_t)chdr[4] | ((uint32_t)chdr[5] << 8) | ((uint32_t)chdr[6] << 16) | ((uint32_t)chdr[7] << 24);

    if (memcmp(chdr, "fmt ", 4) == 0) {
      if (sz < 16) return false;
      uint8_t fmtbuf[32];
      size_t toread = min(sz, (uint32_t)sizeof(fmtbuf));
      if (!readN_http_body(cli, fmtbuf, toread, chunked, chunk_left)) return false;
      uint32_t left = sz - (uint32_t)toread;
      while (left){
        uint8_t dump[64];
        size_t d = min((uint32_t)sizeof(dump), left);
        if (!readN_http_body(cli, dump, d, chunked, chunk_left)) return false;
        left -= d;
      }
      fmt.audioFormat   = (uint16_t) (fmtbuf[0] | (fmtbuf[1] << 8));
      fmt.numChannels   = (uint16_t) (fmtbuf[2] | (fmtbuf[3] << 8));
      fmt.sampleRate    = (uint32_t) (fmtbuf[4] | (fmtbuf[5] << 8) | (fmtbuf[6] << 16) | (fmtbuf[7] << 24));
      fmt.byteRate      = (uint32_t) (fmtbuf[8] | (fmtbuf[9] << 8) | (fmtbuf[10] << 16) | (fmtbuf[11] << 24));
      fmt.blockAlign    = (uint16_t) (fmtbuf[12] | (fmtbuf[13] << 8));
      fmt.bitsPerSample = (uint16_t) (fmtbuf[14] | (fmtbuf[15] << 8));
      gotFmt = true;
    }
    else if (memcmp(chdr, "data", 4) == 0) {
      if (!gotFmt) return false;
      dataRemaining = sz;
      return true;
    }
    else {
      uint32_t left = sz;
      while (left){
        uint8_t dump[128];
        size_t d = min((uint32_t)sizeof(dump), left);
        if (!readN_http_body(cli, dump, d, chunked, chunk_left)) return false;
        left -= d;
      }
    }
  }
}

// ---- HTTP 播放任务
static TaskHandle_t taskHttpPlayHandle = nullptr;
static volatile bool http_play_running = false;

void taskHttpPlay(void*){
  http_play_running = true;
  WiFiClient cli;

  auto readLine = [&](String& out, uint32_t timeout_ms)->bool {
    out = "";
    uint32_t t0 = millis();
    while (millis() - t0 < timeout_ms) {
      while (cli.available()) {
        char c = (char)cli.read();
        if (c == '\r') continue;
        if (c == '\n') return true;
        out += c;
        if (out.length() > 1024) return false;
      }
      delay(1);
    }
    return false;
  };

  auto readNRaw = [&](uint8_t* dst, size_t n, uint32_t timeout_ms)->bool {
    size_t got = 0;
    uint32_t t0 = millis();
    while (got < n) {
      if (!cli.connected()) return false;
      int avail = cli.available();
      if (avail > 0) {
        int take = (int)min((size_t)avail, n - got);
        int r = cli.read(dst + got, take);
        if (r > 0) { got += r; continue; }
      }
      if (millis() - t0 > timeout_ms) return false;
      delay(1);
    }
    return true;
  };

  auto makeBodyReader = [&](bool& is_chunked, uint32_t& chunk_left){
    return [&](uint8_t* dst, size_t n, uint32_t timeout_ms)->bool {
      size_t filled = 0;
      uint32_t t0 = millis();
      while (filled < n) {
        if (!cli.connected()) return false;
        if (is_chunked) {
          if (chunk_left == 0) {
            String szLine;
            if (!readLine(szLine, timeout_ms)) return false;
            int sc = szLine.indexOf(';');
            if (sc >= 0) szLine = szLine.substring(0, sc);
            szLine.trim();
            uint32_t sz = 0;
            if (sscanf(szLine.c_str(), "%x", &sz) != 1) return false;
            if (sz == 0) { String dummy; readLine(dummy, 200); return false; }
            chunk_left = sz;
          }
          size_t need = (size_t)min<uint32_t>(chunk_left, (uint32_t)(n - filled));
          while (cli.available() < (int)need) {
            if (millis() - t0 > timeout_ms) return false;
            if (!cli.connected()) return false;
            delay(1);
          }
          int r = cli.read(dst + filled, need);
          if (r <= 0) {
            if (millis() - t0 > timeout_ms) return false;
            delay(1); continue;
          }
          filled     += r;
          chunk_left -= r;
          if (chunk_left == 0) {
            char crlf[2];
            if (!readNRaw((uint8_t*)crlf, 2, 200)) return false;
          }
        } else {
          if (!readNRaw(dst + filled, n - filled, timeout_ms)) return false;
          filled = n;
        }
      }
      return true;
    };
  };

  static int32_t outLR[1024 * 2];
  const uint32_t BODY_TIMEOUT_MS = 1500;

  while (http_play_running) {
    static unsigned long lastHttpAttempt = 0;
    static unsigned long http_backoff_ms = 1000;
    const unsigned long http_backoff_max = 30000;

    // If WiFi is not connected, skip attempts to create new TCP sockets.
    if (WiFi.status() != WL_CONNECTED) {
      if (millis() - lastHttpAttempt > 1000) {
        lastHttpAttempt = millis();
        Serial.printf("[AUDIO] skip HTTP connect: WiFi not connected (status=%d)\n", WiFi.status());
      }
      delay(500);
      continue;
    }

    if (!cli.connected()) {
      // avoid rapid reconnect attempts which can exhaust sockets/FDs
      if (millis() - lastHttpAttempt < http_backoff_ms) { vTaskDelay(pdMS_TO_TICKS(100)); continue; }
      lastHttpAttempt = millis();
      Serial.println("[AUDIO] HTTP connect...");
      if (!cli.connect(SERVER_HOST, SERVER_PORT)) {
        Serial.printf("[AUDIO] HTTP connect() failed, freeHeap=%u\n", ESP.getFreeHeap());
        cli.stop();
        http_backoff_ms = min(http_backoff_ms * 2, http_backoff_max);
        delay(500);
        continue;
      }
      // reset backoff on success
      http_backoff_ms = 1000;
      String req =
        String("GET /stream.wav HTTP/1.1\r\n") +
        "Host: " + SERVER_HOST + ":" + String(SERVER_PORT) + "\r\n" +
        "Connection: keep-alive\r\n\r\n";
      cli.print(req);
    }

    bool header_ok  = false;
    bool is_chunked = false;
    uint32_t content_len = 0;
    {
      String line; uint32_t t0 = millis();
      while (millis() - t0 < 3000) {
        if (!readLine(line, 1000)) { if (!cli.connected()) break; continue; }
        String u = line; u.toLowerCase();
        if (u.startsWith("transfer-encoding:")) { if (u.indexOf("chunked") >= 0) is_chunked = true; }
        else if (u.startsWith("content-length:")) { content_len = (uint32_t) strtoul(u.substring(strlen("content-length:")).c_str(), nullptr, 10); }
        if (line.length() == 0) { header_ok = true; break; }
      }
    }
    if (!header_ok) { cli.stop(); delay(300); continue; }

    uint32_t chunk_left = 0;
    auto readBody = makeBodyReader(is_chunked, chunk_left);

    uint8_t hdr12[12];
    if (!readBody(hdr12, 12, 1000)) { cli.stop(); delay(300); continue; }
    if (memcmp(hdr12, "RIFF", 4) != 0 || memcmp(hdr12 + 8, "WAVE", 4) != 0) { cli.stop(); delay(300); continue; }

    bool  gotFmt = false, gotData = false;
    uint8_t chdr[8];
    uint16_t audioFormat=0, numChannels=0, bitsPerSample=0;
    uint32_t sampleRate=0;

    while (!gotData) {
      if (!readBody(chdr, 8, 1000)) { cli.stop(); delay(300); goto reconnect; }
      uint32_t sz = (uint32_t)chdr[4] | ((uint32_t)chdr[5]<<8) | ((uint32_t)chdr[6]<<16) | ((uint32_t)chdr[7]<<24);

      if (memcmp(chdr, "fmt ", 4) == 0) {
        if (sz < 16) { cli.stop(); delay(300); goto reconnect; }
        uint8_t fmtbuf[32];
        size_t toread = min(sz, (uint32_t)sizeof(fmtbuf));
        if (!readBody(fmtbuf, toread, 1000)) { cli.stop(); delay(300); goto reconnect; }
        if (sz > toread) {
          size_t left = sz - toread;
          while (left) { uint8_t dump[128]; size_t d = min(left, sizeof(dump));
            if (!readBody(dump, d, 1000)) { cli.stop(); delay(300); goto reconnect; }
            left -= d;
          }
        }
        audioFormat   = (uint16_t)(fmtbuf[0] | (fmtbuf[1] << 8));
        numChannels   = (uint16_t)(fmtbuf[2] | (fmtbuf[3] << 8));
        sampleRate    = (uint32_t)(fmtbuf[4] | (fmtbuf[5] << 8) | (fmtbuf[6] << 16) | (fmtbuf[7] << 24));
        bitsPerSample = (uint16_t)(fmtbuf[14] | (fmtbuf[15] << 8));
        gotFmt = true;
      }
      else if (memcmp(chdr, "data", 4) == 0) {
        if (!gotFmt) { cli.stop(); delay(300); goto reconnect; }
        gotData = true;
      }
      else {
        size_t left = sz;
        while (left) { uint8_t dump[128]; size_t d = min(left, sizeof(dump));
          if (!readBody(dump, d, 1000)) { cli.stop(); delay(300); goto reconnect; }
          left -= d;
        }
      }
    }

    if (!(audioFormat==1 && numChannels==1 && bitsPerSample==16 && (sampleRate==8000 || sampleRate==12000 || sampleRate==16000))) {
      Serial.printf("[AUDIO] unsupported fmt: ch=%u bits=%u sr=%u af=%u\n",
                    numChannels, bitsPerSample, sampleRate, audioFormat);
      cli.stop(); delay(300); continue;
    }
    Serial.printf("[AUDIO] WAV ok: %u/16bit/mono (chunked=%d)\n", sampleRate, is_chunked ? 1 : 0);

    static uint32_t current_out_rate = 0;
    if (current_out_rate != sampleRate) {
      // 重新配置I2S输出采样率以匹配服务端WAV
      i2sOut.begin(I2S_MODE_STD, (int)sampleRate, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO);
      current_out_rate = sampleRate;
      Serial.printf("[I2S OUT] reconfig to %u Hz\n", sampleRate);
    }

    while (http_play_running) {
      uint8_t inbuf[2048];
      size_t  filled = 0;

      // 根据采样率计算20ms字节数（mono,16bit）
      uint32_t bytes20 = (sampleRate * 2 * 20) / 1000; // 16k=640,12k=480,8k=320
      if (bytes20 < 2) bytes20 = 2;

      if (!readBody(inbuf, bytes20, BODY_TIMEOUT_MS)) { break; }
      filled = bytes20;

      while (filled + bytes20 <= sizeof(inbuf)) {
        if (!readBody(inbuf + filled, bytes20, 2)) { break; }
        filled += bytes20;
      }

      if (filled & 1) filled -= 1;
      if (filled == 0) { vTaskDelay(pdMS_TO_TICKS(1)); continue; }

      size_t samp = filled / 2;
      mono16_to_stereo32_msb((const int16_t*)inbuf, samp, outLR, 0.8f);

      size_t bytes = samp * 2 * sizeof(int32_t);
      size_t off = 0;
      while (off < bytes && http_play_running) {
        size_t wrote = i2sOut.write((uint8_t*)outLR + off, bytes - off);
        if (wrote == 0) vTaskDelay(pdMS_TO_TICKS(1));
        else off += wrote;
      }
    }

  reconnect:
    cli.stop();
    delay(200);
  }

  cli.stop();
  vTaskDelete(nullptr);
}

void startStreamWav(){
  if (taskHttpPlayHandle) return;
  xTaskCreatePinnedToCore(taskHttpPlay, "http_wav", 8192, nullptr, 2, &taskHttpPlayHandle, 0);
  Serial.println("[AUDIO] http_wav task started");
}
void stopStreamWav(){
  if (!taskHttpPlayHandle) return;
  http_play_running = false;
  vTaskDelay(pdMS_TO_TICKS(50));
  taskHttpPlayHandle = nullptr;
  Serial.println("[AUDIO] http_wav task stopped");
}

// ====================================================================
// TTS（二进制分片）保留但默认不启用
// ====================================================================
void taskTTSPlay(void*){
  static int32_t stereo32Buf[1024*2];
  for(;;){
    if (!tts_playing){ vTaskDelay(pdMS_TO_TICKS(5)); continue; }
    TTSChunk ch;
    if (xQueueReceive(qTTS, &ch, pdMS_TO_TICKS(50)) == pdPASS){
      size_t inSamp  = ch.n / 2;
      int16_t* inPtr = (int16_t*)ch.data;
      size_t outPairs = 0;
      for (size_t i = 0; i < inSamp; ++i){
        int32_t s = (int32_t)inPtr[i];
        s = (s * 19660) / 32768;
        int32_t v32 = s << 16;
        stereo32Buf[outPairs*2 + 0] = v32;
        stereo32Buf[outPairs*2 + 1] = v32;
        outPairs++;
        if (outPairs >= 1024){
          size_t bytes = outPairs * 2 * sizeof(int32_t);
          size_t off = 0;
          while (off < bytes){
            size_t wrote = i2sOut.write((uint8_t*)stereo32Buf + off, bytes - off);
            if (wrote == 0) vTaskDelay(pdMS_TO_TICKS(1)); else off += wrote;
          }
          outPairs = 0;
        }
      }
      if (outPairs){
        size_t bytes = outPairs * 2 * sizeof(int32_t);
        size_t off = 0;
        while (off < bytes){
          size_t wrote = i2sOut.write((uint8_t*)stereo32Buf + off, bytes - off);
          if (wrote == 0) vTaskDelay(pdMS_TO_TICKS(1)); else off += wrote;
        }
      }
    }
  }
}

inline void tts_reset_queue(){ if (qTTS) xQueueReset(qTTS); }

// ====================================================================
// IMU (ICM42688 over SPI) 50Hz via UDP
// ====================================================================

// --- ICM42688-P registers (Bank0) ---
#define REG_WHO_AM_I      0x75  // expect 0x47
#define REG_BANK_SEL      0x76
#define REG_PWR_MGMT0     0x4E  // 0x0F => accel+gyro LN
#define REG_TEMP_H        0x1D  // then ACC(1F..24), GYR(25..2A)
#define BURST_FIRST       REG_TEMP_H
#define BURST_COUNT       14

// scale (常见默认为 ±16g / ±2000 dps)
static const float ACC_LSB_PER_G   = 2048.0f;   // 1 g = 2048 LSB
static const float GYR_LSB_PER_DPS = 16.4f;     // 1 dps = 16.4 LSB
static const float G               = 9.80665f;
static const float TEMP_SENS       = 132.48f;   // °C/LSB
static const float TEMP_OFFSET     = 25.0f;

static inline void imu_cs_low()  { digitalWrite(IMU_SPI_CS, LOW);  }
static inline void imu_cs_high() { digitalWrite(IMU_SPI_CS, HIGH); }

uint8_t imu_read8(uint8_t reg){
  imu_cs_low();
  SPI.transfer(reg | 0x80);
  uint8_t v = SPI.transfer(0x00);
  imu_cs_high();
  return v;
}
void imu_write8(uint8_t reg, uint8_t val){
  imu_cs_low();
  SPI.transfer(reg & 0x7F);
  SPI.transfer(val);
  imu_cs_high();
}
void imu_readn(uint8_t start_reg, uint8_t* dst, size_t n){
  imu_cs_low();
  SPI.transfer(start_reg | 0x80);
  for (size_t i=0;i<n;i++) dst[i] = SPI.transfer(0x00);
  imu_cs_high();
}

bool imu_init_spi(){
  SPI.begin(IMU_SPI_SCK, IMU_SPI_MISO, IMU_SPI_MOSI, IMU_SPI_CS);
  pinMode(IMU_SPI_CS, OUTPUT);
  imu_cs_high();
  delay(5);

  uint8_t who = imu_read8(REG_WHO_AM_I);
  Serial.printf("[IMU] WHO_AM_I=0x%02X (expect 0x47)\n", who);
  if (who != 0x47) return false;

  imu_write8(REG_PWR_MGMT0, 0x0F); // accel+gyro LN
  delay(10);
  return true;
}

bool imu_read_once(float& tempC, float& ax, float& ay, float& az, float& gx, float& gy, float& gz){
  uint8_t raw[BURST_COUNT];
  imu_readn(BURST_FIRST, raw, sizeof(raw));

  auto s16 = [&](int idx)->int16_t {
    return (int16_t)((raw[idx] << 8) | raw[idx+1]);
  };

  int16_t tr  = s16(0);
  int16_t axr = s16(2);
  int16_t ayr = s16(4);
  int16_t azr = s16(6);
  int16_t gxr = s16(8);
  int16_t gyr = s16(10);
  int16_t gzr = s16(12);

  tempC = (float)tr / TEMP_SENS + TEMP_OFFSET;
  ax = ((float)axr / ACC_LSB_PER_G) * G;
  ay = ((float)ayr / ACC_LSB_PER_G) * G;
  az = ((float)azr / ACC_LSB_PER_G) * G;
  gx =  (float)gxr / GYR_LSB_PER_DPS;
  gy =  (float)gyr / GYR_LSB_PER_DPS;
  gz =  (float)gzr / GYR_LSB_PER_DPS;

  return true;
}

// 轻微平滑，便于观察；不改变 UDP 字段名
static const float EMA_ALPHA = 0.20f;
bool  ema_inited = false;
float ax_f=0, ay_f=0, az_f=0;

void taskImuLoop(void*){
  for(;;){
    static bool inited = false;
    if (!inited){
      inited = imu_init_spi();
      if (!inited){ vTaskDelay(pdMS_TO_TICKS(500)); continue; }
      Serial.println("[IMU] init OK (SPI)");
    }

    float tempC, ax, ay, az, gx, gy, gz;
    if (!imu_read_once(tempC, ax, ay, az, gx, gy, gz)){
      inited = false; vTaskDelay(pdMS_TO_TICKS(50)); continue;
    }

    if (!ema_inited){ ax_f=ax; ay_f=ay; az_f=az; ema_inited=true; }
    else {
      ax_f = EMA_ALPHA*ax + (1-EMA_ALPHA)*ax_f;
      ay_f = EMA_ALPHA*ay + (1-EMA_ALPHA)*ay_f;
      az_f = EMA_ALPHA*az + (1-EMA_ALPHA)*az_f;
    }

    char buf[256];
    unsigned long ts = millis();
    int n = snprintf(buf, sizeof(buf),
      "{\"ts\":%lu,\"temp_c\":%.2f,"
      "\"accel\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f},"
      "\"gyro\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f}}",
      ts, tempC, ax_f, ay_f, az_f, gx, gy, gz);

    if (n > 0) {
      udp.beginPacket(UDP_HOST, UDP_PORT);
      udp.write((const uint8_t*)buf, n);
      udp.endPacket();
    }
    vTaskDelay(pdMS_TO_TICKS(20)); // 50 Hz
  }
}

// ====================================================================
// Setup / Loop
// ====================================================================
void setup() {
  Serial.begin(115200);
  // Some ESP32-S3 USB-Serial/JTAG setups drop initial bytes right after reset.
  // Emit a short heartbeat so we can reliably confirm the sketch is running.
  for (int i = 0; i < 10; i++) {
    delay(200);
    Serial.println(".");
  }
  Serial.println();
  Serial.println("[BOOT] XIAO ESP32S3 firmware start");
  Serial.printf("[BOOT] reset_reason=%d\n", (int)esp_reset_reason());
  Serial.printf("[BOOT] target server ws: ws://%s:%u%s\n", SERVER_HOST, SERVER_PORT, CAM_WS_PATH);
  ets_printf("\n[BOOT] XIAO ESP32S3 firmware start\n");
  ets_printf("[BOOT] reset_reason=%d\n", (int)esp_reset_reason());
  ets_printf("[BOOT] target server ws: ws://%s:%u%s\n", SERVER_HOST, (unsigned)SERVER_PORT, CAM_WS_PATH);
  #if ENABLE_AUDIO
  Serial.printf("[BOOT] target server aud: ws://%s:%u%s\n", SERVER_HOST, SERVER_PORT, AUD_WS_PATH);
  ets_printf("[BOOT] target server aud: ws://%s:%u%s\n", SERVER_HOST, (unsigned)SERVER_PORT, AUD_WS_PATH);
  #endif
  Serial.flush();
  WiFi.onEvent([](WiFiEvent_t event, WiFiEventInfo_t info){
  Serial.printf("[WiFiEvent] %d\n", event);
  });
  delay(300);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] connecting");
  while (WiFi.status()!=WL_CONNECTED){ delay(300); Serial.print("."); }
  Serial.println(" OK " + WiFi.localIP().toString());

  if (!init_camera()) {
    Serial.println("[CAM] init failed (will not reboot-loop). Check camera wiring/pins/board.");
    Serial.flush();
    // Stay alive so serial monitor can capture logs on boards without buttons.
    while (true) {
      delay(1000);
    }
  }

  // Camera warmup: capture a few frames immediately so we can confirm the camera
  // actually produces JPEG buffers on this board/config.
  for (int i = 0; i < 3; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) {
      ets_printf("[CAM-WARM] %d ok len=%u fmt=%d\n", i, (unsigned)fb->len, (int)fb->format);
      esp_camera_fb_return(fb);
    } else {
      ets_printf("[CAM-WARM] %d FAIL\n", i);
    }
    delay(200);
  }

  #if ENABLE_UDP
  udp.begin(0);
  #endif

  #if ENABLE_AUDIO
  init_i2s_in();
  #endif
  #if ENABLE_TTS
  init_i2s_out();
  #endif

  // Match queue depth to camera fb_count to avoid enqueuing more frame pointers
  // than there are underlying buffers.
  const int qFramesDepth = (g_cam_fb_count > 0) ? g_cam_fb_count : 1;
  qFrames = xQueueCreate(qFramesDepth, sizeof(frame_ptr_t));
  #if ENABLE_AUDIO
  qAudio  = xQueueCreate(AUDIO_QUEUE_DEPTH, sizeof(AudioChunk));
  wsAudMutex = xSemaphoreCreateMutex();
  #endif
  #if ENABLE_TTS
  qTTS    = xQueueCreate(TTS_QUEUE_DEPTH, sizeof(TTSChunk));
  #endif

  xTaskCreatePinnedToCore(taskCamCapture, "cam_cap", 10240, NULL, 4, NULL, 1);
  xTaskCreatePinnedToCore(taskWsCamLoop,  "ws_cam",   8192, NULL, 3, NULL, 1);
  #if ENABLE_AUDIO
  xTaskCreatePinnedToCore(taskMicCapture, "mic_cap",   4096, NULL, 2, NULL, 0);
  xTaskCreatePinnedToCore(taskMicUpload,  "mic_upl",   4096, NULL, 2, NULL, 1);
  #endif
  #if ENABLE_IMU
  xTaskCreatePinnedToCore(taskImuLoop,    "imu_loop",  4096, NULL, 2, NULL, 0);
  #endif
  #if ENABLE_TTS
  xTaskCreatePinnedToCore(taskTTSPlay,    "tts_play",  4096, NULL, 2, NULL, 0);
  #endif

  wsCam.onEvent([](WebsocketsEvent ev, String){
    if (ev == WebsocketsEvent::ConnectionOpened)  { 
      cam_ws_ready = true;  
      Serial.println("[WS-CAM] open");
      // 重置统计
      frame_sent_count = 0;
      frame_dropped_count = 0;
      ws_send_fail_count = 0;
      last_stats_time = millis();
    }
    if (ev == WebsocketsEvent::ConnectionClosed)  { 
      cam_ws_ready = false; 
      Serial.printf("[WS-CAM] closed (sent=%lu, dropped=%lu, fail=%lu)\n", 
                    frame_sent_count, frame_dropped_count, ws_send_fail_count);
    }
  });

  wsCam.onMessage([](WebsocketsMessage msg){
    if (msg.isText()){
      String cmd = msg.data(); cmd.trim();
      if (cmd.startsWith("SET:FRAMESIZE=")) {
        String v = cmd.substring(strlen("SET:FRAMESIZE="));
        v.toUpperCase();
        framesize_t fs = g_frame_size;
        if (v == "SVGA") fs = FRAMESIZE_SVGA;
        else if (v == "XGA") fs = FRAMESIZE_XGA;
        else if (v == "VGA") fs = FRAMESIZE_VGA;
        if (apply_framesize(fs)) Serial.printf("[CAM] framesize set to %s\n", v.c_str());
        else Serial.printf("[CAM] framesize set failed: %s\n", v.c_str());
      }
      else if (cmd.startsWith("SET:QUALITY=")) {     // 新增：动态画质
        int q = cmd.substring(strlen("SET:QUALITY=")).toInt();
        q = constrain(q, 5, 40);
        sensor_t* s = esp_camera_sensor_get();
        if (s) { s->set_quality(s, q); Serial.printf("[CAM] quality=%d\n", q); }
      }
      else if (cmd.startsWith("SET:FPS=")) {         // 新增：发送节流FPS
        int f = cmd.substring(strlen("SET:FPS=")).toInt();
        g_target_fps = (f <= 0 ? 0 : constrain(f, 5, 60));
        Serial.printf("[CAM] target_fps=%d\n", g_target_fps);
      }

      else if (cmd == "SNAP:HQ") {
        Serial.println("[CAM] SNAP:HQ request");
        if (snapshot_in_progress) return;
        snapshot_in_progress = true;
        sensor_t* s = esp_camera_sensor_get();
        framesize_t old_fs = g_frame_size;
        int old_q = JPEG_QUALITY;
        // 目标分辨率：XGA（若需更高可改为 SXGA/UXGA，视PSRAM稳定性）
        framesize_t target_fs = FRAMESIZE_SXGA;
        if (s) {
          s->set_framesize(s, target_fs);
          s->set_quality(s, 18); // 数值越小越清晰
        }
        vTaskDelay(pdMS_TO_TICKS(500));
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb && fb->format == PIXFORMAT_JPEG) {
          wsCam.send("SNAP:BEGIN");
          bool ok = wsCam.sendBinary((const char*)fb->buf, fb->len);
          wsCam.send("SNAP:END");
          if (!ok) { Serial.println("[CAM] SNAP send failed"); }
          esp_camera_fb_return(fb);
        } else {
          if (fb) esp_camera_fb_return(fb);
          Serial.println("[CAM] SNAP: capture failed");
        }
        if (s) {
          s->set_framesize(s, old_fs);
          s->set_quality(s, old_q);
        }
        snapshot_in_progress = false;
      }
    }
  });

  #if ENABLE_AUDIO
  wsAud.onEvent([](WebsocketsEvent ev, String){
    if (ev == WebsocketsEvent::ConnectionOpened)  { aud_ws_ready = true;  Serial.println("[WS-AUD] open"); }
    if (ev == WebsocketsEvent::ConnectionClosed)  { 
      aud_ws_ready = false; 
      Serial.println("[WS-AUD] closed"); 
      stopStreamWav();
    }
  });

  wsAud.onMessage([](WebsocketsMessage msg){
    if (msg.isText()){
      String s = msg.data(); s.trim();
      if (s == "RESTART"){
        run_audio_stream = false; xQueueReset(qAudio); delay(50);
        wsAud.send("START"); run_audio_stream = true;
      }
    }
  });
  #endif
}

void loop() {
  // Heartbeat log: if your monitor shows nothing, power-cycle the board (unplug/replug USB)
  // and you should see these lines within 2 seconds.
  {
    static unsigned long lastHb = 0;
    unsigned long now = millis();
    if (now - lastHb > 2000) {
      lastHb = now;

      // Camera WS is owned by taskWsCamLoop; use snapshot vars only.
      int camAvailSnap = (int)g_wsCamAvailSnap;
      int audAvailSnap = -1;
      #if ENABLE_AUDIO
      {
        MutexLock lock(wsAudMutex, 0);
        if (lock.locked) audAvailSnap = (int)wsAud.available();
      }
      #endif

      Serial.printf(
        "[HB] ms=%lu wifi=%d ip=%s cam_avail=%d cam_ready=%d aud_avail=%d aud_ready=%d heap=%u backoff=%lu\n",
        now,
        (int)WiFi.status(),
        WiFi.isConnected() ? WiFi.localIP().toString().c_str() : "0.0.0.0",
        camAvailSnap,
        (int)cam_ws_ready,
        audAvailSnap,
        (int)aud_ws_ready,
        (unsigned)ESP.getFreeHeap(),
        (unsigned long)wsCam_backoff_ms
      );
      Serial.flush();

      // Mirror heartbeat to UART0-level logging so we can always observe state.
      unsigned q = 0;
      if (qFrames) q = (unsigned)uxQueueMessagesWaiting(qFrames);
      ets_printf(
        "[HB] ms=%lu wifi=%d cam_avail=%d cam_ready=%d snap=%d heap=%u backoff=%lu q=%u cap=%lu cap_fail=%lu sent=%lu drop=%lu fail=%lu\n",
        (unsigned long)now,
        (int)WiFi.status(),
        (int)camAvailSnap,
        (int)cam_ws_ready,
        (int)snapshot_in_progress,
        (unsigned)ESP.getFreeHeap(),
        (unsigned long)wsCam_backoff_ms,
        (unsigned)q,
        (unsigned long)frame_captured_count,
        (unsigned long)frame_capture_fail_total,
        (unsigned long)frame_sent_count,
        (unsigned long)frame_dropped_count,
        (unsigned long)ws_send_fail_count
      );
    }
  }

  // If WiFi is not connected, try reconnect. Camera WS is handled in taskWsCamLoop.
  if (WiFi.status() != WL_CONNECTED) {
    static unsigned long lastWifiCheck = 0;
    if (millis() - lastWifiCheck > 2000) {
      lastWifiCheck = millis();
      Serial.printf("[WiFi] status=%d, attempting reconnect...\n", WiFi.status());
      WiFi.reconnect();
    }
    #if ENABLE_AUDIO
    {
      MutexLock lock(wsAudMutex, pdMS_TO_TICKS(10));
      if (lock.locked && wsAud.available()) wsAud.poll();
    }
    #endif
    delay(500);
    return;
  }

  #if ENABLE_AUDIO
  // Audio WS connection/polling (disabled in image-only mode)
  bool audAvail = false;
  {
    MutexLock lock(wsAudMutex, pdMS_TO_TICKS(10));
    if (lock.locked) audAvail = wsAud.available();
  }
  if (!audAvail) {
    if (WiFi.status() != WL_CONNECTED) {
      if (millis() - lastWsAudAttempt > 1000) {
        lastWsAudAttempt = millis();
        Serial.printf("[WS-AUD] skip connect: WiFi not connected (status=%d)\n", WiFi.status());
      }
    }
    else if (millis() - lastWsAudAttempt > 2000) {
      lastWsAudAttempt = millis();
      Serial.println("[WS-AUD] attempting connect...");
      bool ok = false;
      if (WiFi.status() == WL_CONNECTED) {
        MutexLock lock(wsAudMutex, pdMS_TO_TICKS(500));
        if (lock.locked) {
          if (wsAud.available()) { wsAud.close(); vTaskDelay(pdMS_TO_TICKS(50)); }
          ok = wsAud.connect(SERVER_HOST, SERVER_PORT, AUD_WS_PATH);
        }
      }
      if (ok) {
        Serial.println("[WS-AUD] connected");
        delay(50);
        run_audio_stream = true;
        {
          MutexLock lock(wsAudMutex, pdMS_TO_TICKS(100));
          if (lock.locked && wsAud.available()) wsAud.send("START");
        }
        startStreamWav();
      } else {
        Serial.printf("[WS-AUD] connect failed, freeHeap=%u\n", ESP.getFreeHeap());
        {
          MutexLock lock(wsAudMutex, pdMS_TO_TICKS(100));
          if (lock.locked && wsAud.available()) { wsAud.close(); vTaskDelay(pdMS_TO_TICKS(20)); }
        }
      }
    }
  }
  {
    MutexLock lock(wsAudMutex, pdMS_TO_TICKS(10));
    if (lock.locked && wsAud.available()) wsAud.poll();
  }
  #endif
  delay(2);
}
