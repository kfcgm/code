import network
import socket
import time
import sensor, image, json, tf, gc, machine
import lcd

# 初始化 LCD
lcd.init()

# WiFi配置
SSID = "Xiaomi_D296"
KEY = "123456789"
HOST = "192.168.31.190"  # 服务器地址
PORT = 8234

# 模型与图像配置
labels = ["apple", "banana", "orange", "pear", "grape"]
input_size = 96

# 内存与错误处理参数
MIN_FREE_MEM = 20 * 1024       # 20KB 最小内存保护阈值
MAX_ERRORS_BEFORE_REBOOT = 3   # 错误次数超过自动重启
FRAME_TIMEOUT_MS = 30000       # 单帧超时 watchdog
GC_INTERVAL_MS = 10 * 60 * 1000  # 每10分钟执行一次gc

# 初始化摄像头
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320x240
sensor.skip_frames(time=2000)
sensor.set_auto_whitebal(False)

# 加载模型
net = tf.load("fruit_model.tflite", load_to_fb=True)

# 打印内存状态
def print_memory_info():
    free = gc.mem_free()
    alloc = gc.mem_alloc()
    print("[内存] 可用: %d | 已用: %d | 总: %d" % (free, alloc, free + alloc))

# WiFi连接
def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        wlan.connect(SSID, KEY)
        print("连接WiFi中...")
        timeout = 100
        while not wlan.isconnected() and timeout > 0:
            time.sleep_ms(100)
            timeout -= 1
    if wlan.isconnected():
        print("WiFi连接成功:", wlan.ifconfig())
        return True
    else:
        print("WiFi连接失败")
        return False

# 图像转十六进制字符串
def bytes_to_hex_string(data):
    return ''.join('%02X' % b for b in data)

# 主图像传输逻辑
def tcp_image_sender():
    error_count = 0
    last_gc_time = time.ticks_ms()
    last_frame_time = time.ticks_ms()

    while True:
        try:
            client = socket.socket()
            client.connect((HOST, PORT))
            print("连接服务器成功")
            error_count = 0

            while True:
                # 检查WiFi是否仍连接
                wlan = network.WLAN(network.STA_IF)
                if not wlan.isconnected():
                    print("WiFi断开，尝试重连...")
                    wifi_connect()
                    break  # 重新连接socket

                # 内存不足处理
                if gc.mem_free() < MIN_FREE_MEM:
                    print("[警告] 内存过低，清理垃圾回收并暂停")
                    gc.collect()
                    time.sleep(1)
                    continue

                # Watchdog超时检测
                if time.ticks_diff(time.ticks_ms(), last_frame_time) > FRAME_TIMEOUT_MS:
                    print("[错误] 图像处理超时，系统重启")
                    machine.reset()

                # 图像采集
                img = sensor.snapshot()
                lcd.display(img)
                last_frame_time = time.ticks_ms()

                # 中心裁剪用于模型推理
                x = (img.width() - input_size) // 2
                y = (img.height() - input_size) // 2
                roi_img = img.copy((x, y, input_size, input_size))

                # 模型推理
                out = net.classify(roi_img)[0].output()
                max_idx = out.index(max(out))
                label = labels[max_idx]
                confidence = max(out)

                # 显示识别结果
                img.draw_string(0, 0, "%s (%.2f)" % (label, confidence), color=(255, 0, 0))

                # 构造 JSON 识别结果
                blob_info = [{
                    'label': label,
                    'confidence': round(confidence, 2)
                }]
                blob_json = json.dumps(blob_info)

                # 图像压缩 + 十六进制编码
                jpg_data = img.compress(quality=35)
                jpg_hex = bytes_to_hex_string(jpg_data)
                del jpg_data
                gc.collect()

                # 拼接数据并发送
                message = jpg_hex + ";result=" + blob_json + "\n"
                client.send(message.encode())

                print("已发送图像和结果")
                print_memory_info()

                # 定期执行垃圾回收
                if time.ticks_diff(time.ticks_ms(), last_gc_time) > GC_INTERVAL_MS:
                    print("[定时GC] 执行垃圾回收")
                    gc.collect()
                    last_gc_time = time.ticks_ms()

                time.sleep(0.5)

        except Exception as e:
            print("[异常] 连接或发送失败:", e)
            try:
                client.close()
            except:
                pass
            error_count += 1
            if error_count >= MAX_ERRORS_BEFORE_REBOOT:
                print("错误次数过多，系统将重启...")
                time.sleep(2)
                machine.reset()
            print("等待5秒后重试...")
            gc.collect()
            time.sleep(5)

# 启动主程序
if __name__ == "__main__":
    if wifi_connect():
        tcp_image_sender()
    else:
        print("WiFi连接失败，程序结束")
